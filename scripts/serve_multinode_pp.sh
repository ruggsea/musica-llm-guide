#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

# ============================================================================
# Multi-node vLLM serving with Ray Pipeline Parallel on MUSICA
# ============================================================================
#
# Uses Ray to split model layers across nodes (Pipeline Parallel).
# Works for most models. For DeepSeek/Kimi/Mistral-Large-3, use DP+EP instead.
#
# Usage:
#   sbatch -N 2 --job-name maverick scripts/serve_multinode_pp.sh \
#     meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 auto 4
#
# Arguments:
#   $1 = HuggingFace model ID
#   $2 = dtype (auto, bfloat16)
#   $3 = tensor parallel size per node (usually 4)
#
# Access: ssh -NL 8000:<head-node>:8000 user@musica.vsc.ac.at
# ============================================================================

set -eo pipefail

MODEL="${1:?Usage: $0 <model_id> <dtype> <tp>}"
DTYPE="${2:-auto}"
TP="${3:-4}"
PP="${SLURM_NNODES}"

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

# ── Node topology ────────────────────────────────────────────
ALL_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$ALL_NODES" | head -n 1)
HEAD_IP=$(srun -N 1 -n 1 -w ${HEAD_NODE} hostname --ip-address 2>/dev/null | head -1)
WORKER_NODES=$(echo "$ALL_NODES" | tail -n +2)

echo "============================================================"
echo "Model:   $MODEL"
echo "Mode:    PP (TP=$TP, PP=$PP)"
echo "Head:    $HEAD_NODE ($HEAD_IP)"
echo "Workers: $(echo $WORKER_NODES | tr '\n' ' ')"
echo "Start:   $(date)"
echo "============================================================"

# ── Start Ray head ───────────────────────────────────────────
srun -N 1 -n 1 -w ${HEAD_NODE} --gpus-per-task=4 \
  bash -c "
    source /data/fs201045/rl41113/vllm-venv/bin/activate
    ray start --block --head --port=6379 --num-gpus=4 --node-ip-address=${HEAD_IP}
  " &
sleep 15

# ── Start Ray workers ────────────────────────────────────────
for WORKER in $WORKER_NODES; do
    WORKER_IP=$(srun -N 1 -n 1 -w ${WORKER} hostname --ip-address 2>/dev/null | head -1)
    srun -N 1 -n 1 -w ${WORKER} --gpus-per-task=4 \
      bash -c "
        source /data/fs201045/rl41113/vllm-venv/bin/activate
        ray start --block --address=${HEAD_IP}:6379 --num-gpus=4 --node-ip-address=${WORKER_IP}
      " &
done
sleep 25

# ── Launch vLLM ──────────────────────────────────────────────
vllm serve "$MODEL" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --download-dir "$HF_HOME" \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000

# ── Cleanup ──────────────────────────────────────────────────
ray stop 2>/dev/null || true
