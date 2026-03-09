#!/bin/bash
#SBATCH --job-name "test_k2_dp"
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=08:00:00
#SBATCH --output=checks/logs/test_k2_dpep_%j.out
#SBATCH --error=checks/logs/test_k2_dpep_%j.err

# ============================================================================
# 08b_test_kimi_k2_dpep.sh -- Kimi K2 with DP+EP mode (no PP)
# ============================================================================
#
# Model: moonshotai/Kimi-K2-Instruct (1032B MoE, 32B active, FP8)
#
# PREVIOUS ATTEMPT (08_test_multinode_4n.sh) FAILED:
#   TP=4 PP=4 with Ray → KeyError: 'model.layers.46.self_attn.attn'
#   This is a vLLM 0.15.1 bug with DeepSeek V3 architecture + pipeline parallel.
#
# FIX: Use DP+EP mode (data parallel + expert parallel) instead of TP+PP.
#   - Each GPU gets full model replica via expert parallelism
#   - No pipeline stages → avoids the PP layer mapping bug
#   - Launch separate vllm serve processes per node (not Ray)
#   - vLLM recipe: https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.html
#
# Memory: 1032B FP8 = ~1032 GB. With DP+EP across 16 GPUs:
#   Each GPU holds a fraction of experts + shared params ≈ 64-70 GB/GPU
#   Remaining ~24-30 GB for KV cache → max_model_len 4096 is safe
#
# CRITICAL: DP+EP does NOT use Ray. Each node runs its own vllm serve.
# CRITICAL: Requires blobfile package (pip install blobfile)
# ============================================================================

set -euo pipefail

export HF_HOME=/data/fs201045/rl41113/hf-cache
export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export UV_LINK_MODE=copy
VENV="/data/fs201045/rl41113/vllm-venv/bin/activate"
source "$VENV"

MODEL="moonshotai/Kimi-K2-Instruct"
DP_SIZE=16          # Total data-parallel ranks (4 nodes × 4 GPUs)
DP_LOCAL=4          # GPUs per node
PORT=8000
RPC_PORT=29600

echo "Starting Kimi K2 DP+EP test at $(date)"
echo "Nodes: $SLURM_NNODES ($SLURM_JOB_NODELIST)"

# ── Pre-download model weights ────────────────────────────────
echo "[$(date +%H:%M:%S)] Pre-downloading Kimi K2 weights..."
huggingface-cli download "$MODEL" --cache-dir "$HF_HOME" 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] Download complete."

# ── Get node list ─────────────────────────────────────────────
ALL_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$ALL_NODES" | head -n 1)
HEAD_IP=$(srun -N 1 -n 1 -w ${HEAD_NODE} hostname --ip-address 2>/dev/null | head -1)

echo "Head node: ${HEAD_NODE} (${HEAD_IP})"
echo "All nodes: $(echo $ALL_NODES | tr '\n' ' ')"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

t0=$(date +%s)

# ── Launch worker nodes first (headless, background) ──────────
RANK=0
for NODE in $ALL_NODES; do
    if [ "$NODE" = "$HEAD_NODE" ]; then
        RANK=$((RANK + DP_LOCAL))
        continue
    fi

    echo "[$(date +%H:%M:%S)] Launching headless worker on ${NODE} (start_rank=$RANK)"
    srun -N 1 -n 1 -w ${NODE} --gres=gpu:4 \
      bash -c "
        source $VENV
        export HF_HOME=/data/fs201045/rl41113/hf-cache
        export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
        export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        vllm serve $MODEL \
            --headless \
            --data-parallel-start-rank $RANK \
            --trust-remote-code \
            --data-parallel-size $DP_SIZE \
            --data-parallel-size-local $DP_LOCAL \
            --data-parallel-address ${HEAD_IP} \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-expert-parallel \
            --download-dir /data/fs201045/rl41113/hf-cache \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.90
      " &
    RANK=$((RANK + DP_LOCAL))
done

sleep 10

# ── Launch head node (master, serves API) ─────────────────────
echo "[$(date +%H:%M:%S)] Launching master on ${HEAD_NODE} (rank 0)"
srun -N 1 -n 1 -w ${HEAD_NODE} --gres=gpu:4 \
  bash -c "
    source $VENV
    export HF_HOME=/data/fs201045/rl41113/hf-cache
    export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
    export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    vllm serve $MODEL \
        --port $PORT \
        --served-model-name kimi-k2 \
        --trust-remote-code \
        --data-parallel-size $DP_SIZE \
        --data-parallel-size-local $DP_LOCAL \
        --data-parallel-address ${HEAD_IP} \
        --data-parallel-rpc-port $RPC_PORT \
        --enable-expert-parallel \
        --download-dir /data/fs201045/rl41113/hf-cache \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --host 0.0.0.0
  " &
SERVER_PID=$!

echo "[$(date +%H:%M:%S)] Waiting for server (PID $SERVER_PID)..."

# ── Wait for server (up to 60 min — 1TB model) ───────────────
ready=0
for i in $(seq 1 360); do
    sleep 10
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
        ready=1
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  Server process died at iteration $i"
        break
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "  ... waiting ($((i*10))s / 3600s)"
    fi
done

if [ $ready -eq 1 ]; then
    t_load=$(($(date +%s) - t0))
    echo "[$(date +%H:%M:%S)] Server ready after ${t_load}s. Testing generation..."

    response=$(curl -s --max-time 120 http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"kimi-k2","prompt":"Hello, my name is","max_tokens":32,"temperature":0.7}' 2>&1)

    if echo "$response" | grep -q "choices"; then
        t1=$(date +%s)
        echo "[$(date +%H:%M:%S)] Generation successful!"
        echo "  Output: $(echo "$response" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:200])' 2>/dev/null || echo "$response" | head -c 300)"
        echo "  Load+warmup: ${t_load}s, Total: $(($t1 - $t0))s"
        echo ""
        echo "======================================================================"
        echo "RESULT: PASS"
        echo "  Model:     moonshotai/Kimi-K2-Instruct (DP+EP mode)"
        echo "  Load time: ${t_load}s"
        echo "======================================================================"
        echo ">>> PASS: moonshotai/Kimi-K2-Instruct"
    else
        echo "Generation failed: $(echo "$response" | head -c 500)"
        echo ">>> FAIL: moonshotai/Kimi-K2-Instruct (generation error)"
    fi
else
    echo ">>> FAIL: moonshotai/Kimi-K2-Instruct (server failed to start in DP+EP mode)"
fi

echo ""
echo "Finished at $(date)"

# Kill all background processes
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true
