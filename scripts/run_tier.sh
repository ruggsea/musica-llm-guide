#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH -N 1
#SBATCH --time=04:00:00

# ============================================================================
# run_tier.sh -- Universal single-node tier runner
# ============================================================================
#
# Tests all models in a tier config sequentially on one node.
#
# Usage:
#   sbatch --gres=gpu:<N> -o logs/<tier>_%j.out -e logs/<tier>_%j.err \
#     scripts/run_tier.sh configs/tiers/<tier>.conf
#
# Or use the submit helper:
#   scripts/submit_tier.sh configs/tiers/<tier>.conf
# ============================================================================

set -euo pipefail

CONFIG="${1:?Usage: scripts/run_tier.sh <tier.conf>}"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

export HF_HOME=/data/fs201045/rl41113/hf-cache
export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
export UV_LINK_MODE=copy
# CUDA_HOME needed for FP8 MoE models (FlashInfer CUTLASS + DeepGEMM JIT)
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export VLLM_ENGINE_READY_TIMEOUT_S=1800
source /data/fs201045/rl41113/vllm-venv/bin/activate

SCRIPT="$(dirname "$0")/test_model_load.py"

echo "============================================================"
echo "Tier: ${TIER_NAME} (${#MODELS[@]} models, ${TIER_GPUS} GPUs)"
echo "============================================================"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "Node:    $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -4

PASS=0
FAIL=0
TOTAL=0
FAILED_MODELS=""
RESULTS=""

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model dtype tp extra <<< "$entry"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "################################################################"
    echo "# TEST $TOTAL/${#MODELS[@]}: $model (dtype=$dtype, tp=$tp)"
    echo "################################################################"

    if python "$SCRIPT" --model "$model" --dtype "$dtype" --tp "$tp" --max-model-len 4096 $extra; then
        PASS=$((PASS + 1))
        RESULTS="${RESULTS}PASS: $model\n"
        echo ">>> PASS: $model"
    else
        FAIL=$((FAIL + 1))
        FAILED_MODELS="$FAILED_MODELS  - $model\n"
        RESULTS="${RESULTS}FAIL: $model\n"
        echo ">>> FAIL: $model"
    fi
done

echo ""
echo "============================================================"
echo "${TIER_NAME^^} TIER SUMMARY: $PASS passed, $FAIL failed out of $TOTAL"
if [ -n "$FAILED_MODELS" ]; then
    echo "Failed models:"
    echo -e "$FAILED_MODELS"
fi
echo "============================================================"
echo "Finished at $(date)"

# Send notification
MSG="${TIER_NAME} tier: ${PASS}/${TOTAL} passed"
if [ $FAIL -gt 0 ]; then
    MSG="${MSG}, ${FAIL} FAILED"
fi
curl -s -d "$MSG" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
