#!/bin/bash
#SBATCH --job-name "test_4node"
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=08:00:00
#SBATCH --output=checks/logs/test_4node_%j.out
#SBATCH --error=checks/logs/test_4node_%j.err

# ============================================================================
# 08_test_multinode_4n.sh -- 4-Node Model (16x H100 = 1504 GB), TP=4 PP=4
# ============================================================================
#
# Models tested (1):
#   1. moonshotai/Kimi-K2-Instruct -- FP8, 1187 GB → 74.2 GB/GPU
#
# Kimi K2 is a 1,032B MoE model (384 experts, 32B active).
# At FP8: 1032*1*1.15 = 1187 GB, needs 4 nodes (16 GPUs) for 74.2 GB/GPU.
#
# CRITICAL: Multi-node requires --distributed-executor-backend ray
# CRITICAL: ray start needs --num-gpus=4 on SLURM (auto-detect fails)
# CRITICAL: --download-dir required so Ray workers find the HF cache
# CRITICAL: Pre-download weights before serving (avoid truncated reads)
# Budget 60 min wait (1TB+ model download + load).
# ============================================================================

set -euo pipefail

export HF_HOME=/data/fs201045/rl41113/hf-cache
export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export UV_LINK_MODE=copy
VENV="/data/fs201045/rl41113/vllm-venv/bin/activate"
source "$VENV"

echo "Starting 4-node Kimi K2 test at $(date)"
echo "Nodes: $SLURM_NNODES ($SLURM_JOB_NODELIST)"

# ── Pre-download model weights ────────────────────────────────
echo "[$(date +%H:%M:%S)] Pre-downloading Kimi K2 weights..."
huggingface-cli download "moonshotai/Kimi-K2-Instruct" --cache-dir "$HF_HOME" 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] Download complete."

# ── Network setup ─────────────────────────────────────────────
HEAD_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
WORKER_HOSTNAMES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2)
HEAD_IP=$(srun -N 1 -n 1 -w ${HEAD_HOSTNAME} hostname --ip-address 2>/dev/null | head -1)
RAY_PORT=6379

echo "Head: ${HEAD_HOSTNAME} (${HEAD_IP})"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
# DO NOT set VLLM_HOST_IP -- it propagates to Ray workers and makes them
# all report the same IP, causing "Every node should have a unique IP" error.

# ── Start Ray cluster (with --num-gpus=4 on each node!) ──────
echo "[$(date +%H:%M:%S)] Starting Ray head on ${HEAD_HOSTNAME} (--num-gpus=4)"
srun -J "ray-head" -N 1 -n 1 -w ${HEAD_HOSTNAME} --gres=gpu:4 \
  bash -c "
    source $VENV
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export HF_HOME=/data/fs201045/rl41113/hf-cache
    ray start --block --head --port=${RAY_PORT} --num-gpus=4 --node-ip-address=${HEAD_IP}
  " &
sleep 15

echo "[$(date +%H:%M:%S)] Starting Ray workers (--num-gpus=4)"
for WORKER in $WORKER_HOSTNAMES; do
    WORKER_IP=$(srun -N 1 -n 1 -w ${WORKER} hostname --ip-address 2>/dev/null | head -1)
    srun -J "ray-worker" -N 1 -n 1 -w ${WORKER} --gres=gpu:4 \
      bash -c "
        source $VENV
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export HF_HOME=/data/fs201045/rl41113/hf-cache
        ray start --block --address=${HEAD_IP}:${RAY_PORT} --num-gpus=4 --node-ip-address=${WORKER_IP}
      " &
done
sleep 50

# ── Verify Ray sees all GPUs ─────────────────────────────────
echo "[$(date +%H:%M:%S)] Verifying Ray cluster..."
python3 -c "
import ray; ray.init(address='${HEAD_IP}:${RAY_PORT}')
r = ray.cluster_resources(); n = [x for x in ray.nodes() if x['Alive']]
print(f'  GPUs: {r.get(\"GPU\",0)}, Nodes: {len(n)}')
for x in n: print(f'    {x[\"NodeManagerAddress\"]}: {x[\"Resources\"].get(\"GPU\",0)} GPUs')
ray.shutdown()
" 2>/dev/null || echo "  WARNING: Could not verify Ray cluster"

# ── Test ──────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "# TEST: moonshotai/Kimi-K2-Instruct (FP8, tp=4, pp=4)"
echo "################################################################"

t0=$(date +%s)

# CRITICAL: --download-dir propagates cache path to Ray workers.
vllm serve "moonshotai/Kimi-K2-Instruct" \
    --dtype auto \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 4 \
    --distributed-executor-backend ray \
    --download-dir "$HF_HOME" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 &
SERVER_PID=$!

echo "[$(date +%H:%M:%S)] Waiting for server (PID $SERVER_PID)..."

ready=0
for i in $(seq 1 360); do  # up to 60 min -- 1TB model download
    sleep 10
    # Check HTTP status code, not body (vLLM /health returns empty 200)
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ]; then
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

    response=$(curl -s --max-time 120 http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"moonshotai/Kimi-K2-Instruct","prompt":"Hello, my name is","max_tokens":32,"temperature":0.7}' 2>&1)

    if echo "$response" | grep -q "choices"; then
        t1=$(date +%s)
        echo "[$(date +%H:%M:%S)] Generation successful!"
        echo "  Output: $(echo "$response" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:200])' 2>/dev/null || echo "$response" | head -c 300)"
        echo "  Load+warmup: ${t_load}s, Total: $(($t1 - $t0))s"
        echo ""
        echo "======================================================================"
        echo "RESULT: PASS"
        echo "  Model:     moonshotai/Kimi-K2-Instruct"
        echo "  Load time: ${t_load}s"
        echo "======================================================================"
        echo ">>> PASS: moonshotai/Kimi-K2-Instruct"
    else
        echo "Generation failed: $(echo "$response" | head -c 500)"
        echo ">>> FAIL: moonshotai/Kimi-K2-Instruct (generation error)"
    fi
else
    echo ">>> FAIL: moonshotai/Kimi-K2-Instruct (server failed to start)"
fi

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "Finished at $(date)"
ray stop 2>/dev/null
