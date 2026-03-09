#!/bin/bash
#SBATCH --job-name "test_xl"
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=04:00:00
#SBATCH --output=checks/logs/test_xl_%j.out
#SBATCH --error=checks/logs/test_xl_%j.err

# ============================================================================
# 04_test_xl.sh -- XL Models (70B class), 4 GPUs / 1 node, BF16 TP=4
# ============================================================================
#
# Models tested (5):
#   1. meta-llama/Llama-3.3-70B-Instruct      -- representative 70B
#   2. deepseek-ai/DeepSeek-R1-Distill-Llama-70B -- EDGE: reasoning distill
#   3. LLM360/K2-V2-Instruct                  -- EDGE: fully open 70B
#   4. Qwen/Qwen3-Next-80B-A3B-Instruct       -- EDGE: next-gen MoE architecture
#   5. Qwen/Qwen3-Next-80B-A3B-Thinking       -- EDGE: thinking/reasoning variant
#
# All 70B dense models: ~40 GB/GPU at BF16 TP=4, leaving ~54 GB headroom per GPU.
# Qwen3-Next-80B: 80B total MoE, BF16 = 184 GB → 46 GB/GPU @ TP=4.
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

echo "Starting XL model tests at $(date)"
nvidia-smi

# --- Representative 70B ---
run_test "meta-llama/Llama-3.3-70B-Instruct" "bfloat16" 4

# --- Edge: R1 reasoning distilled into 70B Llama ---
run_test "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" "bfloat16" 4

# --- Edge: fully open 70B (LLM360/MBZUAI) ---
run_test "LLM360/K2-V2-Instruct" "bfloat16" 4

# --- Edge: next-gen MoE (80B total, 3B active) -- Instruct variant ---
run_test "Qwen/Qwen3-Next-80B-A3B-Instruct" "bfloat16" 4

# --- Edge: next-gen MoE -- Thinking/reasoning variant ---
run_test "Qwen/Qwen3-Next-80B-A3B-Thinking" "bfloat16" 4

echo ""
echo "================================================================"
echo "XL MODELS SUMMARY: $PASS passed, $FAIL failed out of $TOTAL"
if [ -n "$FAILED_MODELS" ]; then
    echo "Failed models:"
    echo -e "$FAILED_MODELS"
fi
echo "================================================================"
echo "Finished at $(date)"
