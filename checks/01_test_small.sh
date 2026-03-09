#!/bin/bash
#SBATCH --job-name "test_small"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=01:00:00
#SBATCH --output=checks/logs/test_small_%j.out
#SBATCH --error=checks/logs/test_small_%j.err

# ============================================================================
# 01_test_small.sh -- Small Models (1-4B), 1 GPU, BF16
# ============================================================================
#
# Models tested (5):
#   1. Qwen/Qwen3-4B              -- representative Qwen small
#   2. google/gemma-3n-E2B-it      -- EDGE: multimodal, needs timm + --trust-remote-code
#   3. google/gemma-3-1b-it        -- EDGE: Gemma 3 architecture, needs --trust-remote-code
#   4. mistralai/Ministral-3-3B-Instruct-2512  -- EDGE: ships FP8 natively
#   5. microsoft/Phi-4-mini-instruct           -- EDGE: 128K context model
#
# All fit comfortably on 1x H100 94GB at BF16 (largest is ~12.5 GB).
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
    local tp="${3:-1}"
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

echo "Starting small model tests at $(date)"
nvidia-smi

# --- Representative ---
run_test "Qwen/Qwen3-4B" "bfloat16"

# --- Edge: multimodal, needs timm library + trust-remote-code ---
run_test "google/gemma-3n-E2B-it" "bfloat16" "1" "--trust-remote-code"

# --- Edge: Gemma 3 architecture, trust-remote-code ---
run_test "google/gemma-3-1b-it" "bfloat16" "1" "--trust-remote-code"

# --- Edge: ships FP8 natively, auto dtype should pick it up ---
run_test "mistralai/Ministral-3-3B-Instruct-2512" "auto"

# --- Edge: 128K context model (testing at 4K) ---
run_test "microsoft/Phi-4-mini-instruct" "bfloat16"

echo ""
echo "================================================================"
echo "SMALL MODELS SUMMARY: $PASS passed, $FAIL failed out of $TOTAL"
if [ -n "$FAILED_MODELS" ]; then
    echo "Failed models:"
    echo -e "$FAILED_MODELS"
fi
echo "================================================================"
echo "Finished at $(date)"
