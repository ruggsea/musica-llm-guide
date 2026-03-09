#!/bin/bash
#SBATCH --job-name "test_medium"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=03:00:00
#SBATCH --output=checks/logs/test_medium_%j.out
#SBATCH --error=checks/logs/test_medium_%j.err

# ============================================================================
# 02_test_medium.sh -- Medium Models (7-14B), 1 GPU, BF16
# ============================================================================
#
# Models tested (9):
#   1. Qwen/Qwen3-8B               -- representative Qwen medium
#   2. Qwen/Qwen3-14B              -- representative, larger medium
#   3. google/gemma-3-12b-it        -- EDGE: multimodal, Gemma arch, --trust-remote-code
#   4. mistralai/Ministral-3-8B-Instruct-2512  -- EDGE: FP8 native, multimodal/vision
#   5. mistralai/Mistral-Nemo-Instruct-2407    -- EDGE: 12B Mistral architecture
#   6. nvidia/NVIDIA-Nemotron-Nano-9B-v2       -- EDGE: hybrid Mamba+Transformer
#   7. microsoft/Phi-4-reasoning-plus           -- EDGE: reasoning variant
#   8. microsoft/Phi-4-mini-flash-reasoning     -- EDGE: needs --trust-remote-code
#   9. internlm/internlm3-8b-instruct          -- EDGE: needs --trust-remote-code
#
# All fit on 1x H100 94GB at BF16 (largest is ~32.2 GB for 14B models).
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

echo "Starting medium model tests at $(date)"
nvidia-smi

# --- Representatives ---
run_test "Qwen/Qwen3-8B" "bfloat16"
run_test "Qwen/Qwen3-14B" "bfloat16"

# --- Edge: multimodal Gemma 3 12B, needs --trust-remote-code ---
run_test "google/gemma-3-12b-it" "bfloat16" "1" "--trust-remote-code"

# --- Edge: Ministral 8B ships FP8, multimodal/vision encoder ---
run_test "mistralai/Ministral-3-8B-Instruct-2512" "auto"

# --- Edge: Mistral Nemo 12B architecture ---
run_test "mistralai/Mistral-Nemo-Instruct-2407" "bfloat16"

# --- Edge: hybrid Mamba+Transformer (Nemotron), needs --trust-remote-code ---
run_test "nvidia/NVIDIA-Nemotron-Nano-9B-v2" "bfloat16" "1" "--trust-remote-code"

# --- Edge: reasoning variant of Phi-4 ---
run_test "microsoft/Phi-4-reasoning-plus" "bfloat16"

# --- Edge: Phi-4 mini flash reasoning, needs --trust-remote-code ---
run_test "microsoft/Phi-4-mini-flash-reasoning" "bfloat16" "1" "--trust-remote-code"

# --- Edge: InternLM3 needs --trust-remote-code ---
run_test "internlm/internlm3-8b-instruct" "bfloat16" "1" "--trust-remote-code"

echo ""
echo "================================================================"
echo "MEDIUM MODELS SUMMARY: $PASS passed, $FAIL failed out of $TOTAL"
if [ -n "$FAILED_MODELS" ]; then
    echo "Failed models:"
    echo -e "$FAILED_MODELS"
fi
echo "================================================================"
echo "Finished at $(date)"
