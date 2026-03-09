#!/bin/bash
#SBATCH --job-name "test_2node"
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=08:00:00
#SBATCH --output=checks/logs/test_2node_%j.out
#SBATCH --error=checks/logs/test_2node_%j.err

# ============================================================================
# 06_test_multinode_2n.sh -- 2-Node Models (8x H100 = 752 GB), TP=4 PP=2
# ============================================================================
#
# Models tested (2):
#   1. Qwen/Qwen3-235B-A22B                    -- BF16, 540 GB → 67.6 GB/GPU
#   2. meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 -- FP8, 460 GB → 57.5 GB/GPU
# GLM-4.5 moved to 3-node (ships BF16, 816 GB = 102 GB/GPU, OOM on 2 nodes)
#
# CRITICAL: Multi-node requires --distributed-executor-backend ray
# CRITICAL: ray start needs --num-gpus=4 on SLURM (auto-detect fails)
# CRITICAL: Pre-download weights before serving (avoid truncated reads)
# Uses vllm serve + curl health check (not test_model_load.py)
# ============================================================================

set -euo pipefail

export HF_HOME=/data/fs201045/rl41113/hf-cache
export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export UV_LINK_MODE=copy
VENV="/data/fs201045/rl41113/vllm-venv/bin/activate"
source "$VENV"

echo "Starting 2-node multi-node tests at $(date)"
echo "Nodes: $SLURM_NNODES ($SLURM_JOB_NODELIST)"

# ── Pre-download all model weights ────────────────────────────
# Avoids truncated safetensors reads when workers load mid-download
echo "[$(date +%H:%M:%S)] Pre-downloading model weights..."
for m in "Qwen/Qwen3-235B-A22B" "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"; do
    echo "  Downloading $m..."
    huggingface-cli download "$m" --cache-dir "$HF_HOME" 2>&1 | tail -1
done
echo "[$(date +%H:%M:%S)] All downloads complete."

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
sleep 25

# ── Verify Ray sees all GPUs ─────────────────────────────────
echo "[$(date +%H:%M:%S)] Verifying Ray cluster..."
python3 -c "
import ray; ray.init(address='${HEAD_IP}:${RAY_PORT}')
r = ray.cluster_resources(); n = [x for x in ray.nodes() if x['Alive']]
print(f'  GPUs: {r.get(\"GPU\",0)}, Nodes: {len(n)}')
for x in n: print(f'    {x[\"NodeManagerAddress\"]}: {x[\"Resources\"].get(\"GPU\",0)} GPUs')
ray.shutdown()
" 2>/dev/null || echo "  WARNING: Could not verify Ray cluster"

# ── Test function ─────────────────────────────────────────────
PASS=0
FAIL=0
TOTAL=0
FAILED_MODELS=""

run_multinode_test() {
    local model="$1"
    local dtype="${2:-bfloat16}"
    local tp="${3:-4}"
    local pp="${4:-2}"
    local max_len="${5:-4096}"
    local extra_args="${6:-}"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "################################################################"
    echo "# TEST $TOTAL: $model (dtype=$dtype, tp=$tp, pp=$pp)"
    echo "################################################################"

    local t0=$(date +%s)

    # CRITICAL: --download-dir propagates cache path to Ray workers.
    # Without it, workers default to ~/.cache/huggingface/ and fail with SafetensorError.
    vllm serve "$model" \
        --dtype "$dtype" \
        --tensor-parallel-size "$tp" \
        --pipeline-parallel-size "$pp" \
        --distributed-executor-backend ray \
        --download-dir "$HF_HOME" \
        --max-model-len "$max_len" \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 8000 \
        $extra_args &
    local SERVER_PID=$!

    echo "[$(date +%H:%M:%S)] Waiting for server (PID $SERVER_PID)..."

    # Wait up to 30 minutes for server
    local ready=0
    for i in $(seq 1 180); do
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
            echo "  ... waiting ($((i*10))s / 1800s)"
        fi
    done

    if [ $ready -eq 1 ]; then
        local t_load=$(($(date +%s) - t0))
        echo "[$(date +%H:%M:%S)] Server ready after ${t_load}s. Testing generation..."

        local response=$(curl -s --max-time 120 http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"'"$model"'","prompt":"Hello, my name is","max_tokens":32,"temperature":0.7}' 2>&1)

        if echo "$response" | grep -q "choices"; then
            local t1=$(date +%s)
            echo "[$(date +%H:%M:%S)] Generation successful!"
            echo "  Output: $(echo "$response" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:200])' 2>/dev/null || echo "$response" | head -c 300)"
            echo "  Load+warmup: ${t_load}s, Total: $(($t1 - $t0))s"
            echo ""
            echo "======================================================================"
            echo "RESULT: PASS"
            echo "  Model:     $model"
            echo "  Load time: ${t_load}s"
            echo "======================================================================"
            PASS=$((PASS + 1))
            echo ">>> PASS: $model"
        else
            echo "Generation failed: $(echo "$response" | head -c 500)"
            FAIL=$((FAIL + 1))
            FAILED_MODELS="$FAILED_MODELS  - $model\n"
            echo ">>> FAIL: $model (generation error)"
        fi
    else
        FAIL=$((FAIL + 1))
        FAILED_MODELS="$FAILED_MODELS  - $model\n"
        echo ">>> FAIL: $model (server failed to start)"
    fi

    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 15
}

# ── Run tests ─────────────────────────────────────────────────

# Qwen3-235B-A22B at BF16: 235*2*1.15 = 540 GB → 540/8 = 67.6 GB/GPU
run_multinode_test "Qwen/Qwen3-235B-A22B" "bfloat16" 4 2 4096

# GLM-4.5: REMOVED from 2-node test. Ships BF16 (not FP8), 355B*2*1.15 = 816 GB = 102 GB/GPU OOM.
# Moved to 3-node test (07) where 816/12 = 68 GB/GPU fits.

# Maverick FP8 checkpoint: 400*1*1.15 = 460 GB → 460/8 = 57.5 GB/GPU
run_multinode_test "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" "auto" 4 2 4096

echo ""
echo "================================================================"
echo "2-NODE SUMMARY: $PASS passed, $FAIL failed out of $TOTAL"
if [ -n "$FAILED_MODELS" ]; then
    echo "Failed models:"
    echo -e "$FAILED_MODELS"
fi
echo "================================================================"
echo "Finished at $(date)"

ray stop 2>/dev/null
