#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

# ============================================================================
# Multi-node vLLM serving with DP+EP (Data Parallel + Expert Parallel)
# ============================================================================
#
# For MoE models where PP is broken or DP+EP gives better throughput:
# DeepSeek-V3/V3.1/V3.2, Kimi-K2/K2.5/K2-Thinking, Mistral-Large-3,
# Step-3.5-Flash, GLM-4.7/5, Qwen3.5-397B
#
# No Ray needed. Launches headless workers on non-head nodes, master on head.
#
# Usage:
#   sbatch -N 3 --job-name deepseek-v32 scripts/serve_multinode_dpep.sh \
#     deepseek-ai/DeepSeek-V3.2 auto
#
# Arguments:
#   $1 = HuggingFace model ID
#   $2 = dtype (auto, bfloat16)
#
# Prerequisites for FP8 MoE:
#   - Pre-compiled FlashInfer CUTLASS kernels (see README)
#   - DeepGEMM with nvcc 12.9 + cuobjdump
# ============================================================================

set -eo pipefail

MODEL="${1:?Usage: $0 <model_id> [dtype]}"
DTYPE="${2:-auto}"
DP_LOCAL=4
DP_SIZE=$(( SLURM_NNODES * DP_LOCAL ))

# ── Environment ──────────────────────────────────────────────
source /data/fs201045/rl41113/vllm-venv/bin/activate
export HF_HOME=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

RPC_PORT=29600

# ── Node topology ────────────────────────────────────────────
ALL_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$ALL_NODES" | head -n 1)
HEAD_IP=$(srun -N 1 -n 1 -w ${HEAD_NODE} hostname --ip-address 2>/dev/null | head -1)
WORKER_NODES=$(echo "$ALL_NODES" | tail -n +2)

echo "============================================================"
echo "Model:   $MODEL"
echo "Mode:    DP+EP (dp_size=$DP_SIZE, dp_local=$DP_LOCAL)"
echo "Head:    $HEAD_NODE ($HEAD_IP)"
echo "Workers: $(echo $WORKER_NODES | tr '\n' ' ')"
echo "Start:   $(date)"
echo "============================================================"

# Common env for srun workers
WORKER_ENV="
    source /data/fs201045/rl41113/vllm-venv/bin/activate
    export HF_HOME=/data/fs201045/rl41113/hf-cache
    export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
    export PATH=\$CUDA_HOME/bin:\$PATH
    export LIBRARY_PATH=\$CUDA_HOME/targets/x86_64-linux/lib/stubs:\${LIBRARY_PATH:-}
"

# ── Launch headless workers ──────────────────────────────────
RANK=0
for NODE in $ALL_NODES; do
    if [ "$NODE" = "$HEAD_NODE" ]; then
        RANK=$((RANK + DP_LOCAL))
        continue
    fi
    echo "[$(date +%H:%M:%S)] Worker on ${NODE} (start_rank=$RANK)"
    srun -N 1 -n 1 -w ${NODE} --gpus-per-task=4 \
      bash -c "
        ${WORKER_ENV}
        vllm serve $MODEL \
            --headless \
            --data-parallel-start-rank $RANK \
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

# ── Launch master ────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Master on ${HEAD_NODE}"
srun -N 1 -n 1 -w ${HEAD_NODE} --gpus-per-task=4 \
  bash -c "
    ${WORKER_ENV}
    vllm serve $MODEL \
        --port 8000 \
        --host 0.0.0.0 \
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
  "

# ── Cleanup ──────────────────────────────────────────────────
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true
