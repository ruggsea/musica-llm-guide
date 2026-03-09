#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH -N 1 --gres=gpu:4
#SBATCH --time=06:00:00

# ============================================================================
# Single-node vLLM serving on MUSICA
# ============================================================================
#
# Usage:
#   sbatch --job-name llama70b scripts/serve_single_node.sh \
#     meta-llama/Llama-3.3-70B-Instruct bfloat16 4
#
# Arguments:
#   $1 = HuggingFace model ID
#   $2 = dtype (bfloat16, auto, float16)
#   $3 = tensor parallel size (1, 2, or 4)
#
# Access the API via SSH tunnel:
#   ssh -NL 8000:<node>:8000 user@musica.vsc.ac.at
#   curl http://localhost:8000/v1/models
# ============================================================================

set -eo pipefail

MODEL="${1:?Usage: $0 <model_id> <dtype> <tp>}"
DTYPE="${2:-bfloat16}"
TP="${3:-4}"

# ── Environment ──────────────────────────────────────────────
# Adjust these paths to your project
source /data/fs201045/rl41113/vllm-venv/bin/activate
export HF_HOME=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# CUDA_HOME for FP8 MoE models (FlashInfer CUTLASS + DeepGEMM JIT)
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

echo "============================================================"
echo "Model: $MODEL"
echo "dtype: $DTYPE | TP: $TP"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"

vllm serve "$MODEL" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000
