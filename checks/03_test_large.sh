#!/bin/bash
#SBATCH --job-name "test_large"
#SBATCH --gres=gpu:2
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=03:00:00
#SBATCH --output=checks/logs/test_large_%j.out
#SBATCH --error=checks/logs/test_large_%j.err

# ============================================================================
# 03_test_large.sh -- Large Models (27-40B), 1-2 GPUs, BF16
# ============================================================================
#
# Models tested (8):
#   1. Qwen/Qwen3-32B              -- representative 32B (TP=2, 36.8 GB/GPU)
#   2. allenai/Olmo-3.1-32B-Instruct         -- EDGE: fully open 32B (TP=2)
#   3. mistralai/Magistral-Small-2509         -- 24B reasoning model (TP=1)
#   4. mistralai/Mistral-Small-3.1-24B-Instruct-2503 -- EDGE: multimodal 24B (TP=1)
#   5. google/gemma-2-27b-it                  -- EDGE: KNOWN ISSUE softcapping (TP=1)
#   6. google/gemma-3-27b-it                  -- EDGE: Gemma 3, --trust-remote-code (TP=1)
#   7. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 -- EDGE: hybrid Mamba+Transformer MoE
#   8. Qwen/Qwen3-Coder-30B-A3B-Instruct     -- EDGE: coding MoE (30B, TP=2)
#
# 32B models: TP=2 recommended (73.6 GB BF16 tight on 1x 94GB)
# 24-27B models: TP=1 fine (55-62 GB at BF16)
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

echo "Starting large model tests at $(date)"
nvidia-smi

# --- Representative 32B at TP=2 ---
run_test "Qwen/Qwen3-32B" "bfloat16" 2

# --- Edge: fully open 32B (Ai2), TP=2 ---
run_test "allenai/Olmo-3.1-32B-Instruct" "bfloat16" 2

# --- 24B reasoning model, fits on 1 GPU (55.2 GB) ---
run_test "mistralai/Magistral-Small-2509" "bfloat16" 1

# --- Edge: multimodal 24B with vision encoder ---
run_test "mistralai/Mistral-Small-3.1-24B-Instruct-2503" "bfloat16" 1

# --- Edge: KNOWN ISSUE -- gemma-2-27b-it fails with flash-attn lacking tanh softcapping
#     Expected to FAIL unless flash-attn is rebuilt with softcapping support.
#     Documenting this as a known issue.
run_test "google/gemma-2-27b-it" "bfloat16" 1

# --- Edge: Gemma 3 27B, needs --trust-remote-code, may need FLASHINFER backend ---
export VLLM_ATTENTION_BACKEND=FLASHINFER
run_test "google/gemma-3-27b-it" "bfloat16" 1 "--trust-remote-code"
unset VLLM_ATTENTION_BACKEND

# --- Edge: NVIDIA hybrid Mamba+Transformer MoE (30B total, 3B active) ---
run_test "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" "bfloat16" 1 "--trust-remote-code"

# --- Edge: coding MoE (30.5B total, 3B active), TP=2 for headroom ---
run_test "Qwen/Qwen3-Coder-30B-A3B-Instruct" "bfloat16" 2

echo ""
echo "================================================================"
echo "LARGE MODELS SUMMARY: $PASS passed, $FAIL failed out of $TOTAL"
if [ -n "$FAILED_MODELS" ]; then
    echo "Failed models:"
    echo -e "$FAILED_MODELS"
fi
echo "================================================================"
echo "Finished at $(date)"
