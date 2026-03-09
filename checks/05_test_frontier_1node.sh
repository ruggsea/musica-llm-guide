#!/bin/bash
#SBATCH --job-name "test_front"
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=06:00:00
#SBATCH --output=checks/logs/test_frontier_%j.out
#SBATCH --error=checks/logs/test_frontier_%j.err

# ============================================================================
# 05_test_frontier_1node.sh -- Frontier MoE Models, 4 GPUs / 1 node
# ============================================================================
#
# Models tested (3):
#   1. meta-llama/Llama-4-Scout-17B-16E-Instruct  -- BF16, 109B MoE, 62.7 GB/GPU
#   2. zai-org/GLM-4.5-Air                        -- BF16, 106B MoE, 60.9 GB/GPU
#   3. XiaomiMiMo/MiMo-V2-Flash                   -- FP8, 309B MoE, 88.8 GB/GPU (tight!)
#
# These are the largest models that fit on 1 node at reasonable precision.
# MiMo-V2-Flash is the tightest fit (~5 GB headroom per GPU).
# ============================================================================

set -euo pipefail

export HF_HOME=/data/fs201045/rl41113/hf-cache
export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
export UV_LINK_MODE=copy
source /data/fs201045/rl41113/vllm-venv/bin/activate

SCRIPT="$(dirname "$0")/test_model_load.py"
PASS=0
FAIL=0
TOTAL=0
FAILED_MODELS=""

run_test() {
    local model="$1"
    local dtype="${2:-auto}"
    local tp="${3:-4}"
    local extra="${4:-}"
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "################################################################"
    echo "# TEST $TOTAL: $model (dtype=$dtype, tp=$tp)"
    echo "################################################################"
    if python "$SCRIPT" --model "$model" --dtype "$dtype" --tp "$tp" --max-model-len 4096 $extra; then
        PASS=$((PASS + 1))
        echo ">>> PASS: $model"
    else
        FAIL=$((FAIL + 1))
        FAILED_MODELS="$FAILED_MODELS  - $model\n"
        echo ">>> FAIL: $model"
    fi
}

echo "Starting frontier 1-node model tests at $(date)"
nvidia-smi

# --- Llama 4 Scout: 109B MoE (16 experts), BF16 fits (62.7 GB/GPU) ---
run_test "meta-llama/Llama-4-Scout-17B-16E-Instruct" "bfloat16" 4

# --- GLM-4.5-Air: 106B MoE, BF16 fits (60.9 GB/GPU), needs --trust-remote-code ---
run_test "zai-org/GLM-4.5-Air" "bfloat16" 4 "--trust-remote-code"

# --- MiMo-V2-Flash: 309B MoE, ships FP8 (88.8 GB/GPU, tight!) ---
# Uses --trust-remote-code for custom architecture
run_test "XiaomiMiMo/MiMo-V2-Flash" "auto" 4 "--trust-remote-code"

echo ""
echo "================================================================"
echo "FRONTIER 1-NODE SUMMARY: $PASS passed, $FAIL failed out of $TOTAL"
if [ -n "$FAILED_MODELS" ]; then
    echo "Failed models:"
    echo -e "$FAILED_MODELS"
fi
echo "================================================================"
echo "Finished at $(date)"
