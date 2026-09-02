#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time=01:30:00

# Universal single-node vllm-serve smoke test (conf-driven).
# Usage: sbatch --job-name <name> -o logs/single2026/<name>_%j.out -e ... \
#          scripts/run_smoke_single.sh configs/single2026/<model>.conf
set -uo pipefail
CONFIG="${1:?usage: run_smoke_single.sh <conf>}"
source "$CONFIG"

# --- abnormal-termination notifier (preemption, node failure, walltime kill) ---
RESULT="${RESULT:-}"
_notify_abnormal(){ if [ -z "${RESULT:-}" ] || [ "${RESULT:-}" = "" ]; then
  curl -s -H "Title: MUSICA guide: ABORTED" -d "ABORTED (no result): ${SERVED_NAME:-job} slurm=${SLURM_JOB_ID:-?} — preempted/killed/node-fail" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
fi; }
trap _notify_abnormal EXIT


export HF_HOME=${GUIDE_HF_HOME:-/data/fs201045/rl41113/hf-cache}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export XDG_CACHE_HOME=/data/fs201045/rl41113/xdg-cache
mkdir -p $XDG_CACHE_HOME
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
[ "${NEEDS_DEEPGEMM:-false}" = "true" ] && export VLLM_USE_DEEP_GEMM=1
source "${VENV:-/data/fs201045/rl41113/vllm-venv}/bin/activate"

TRUST_FLAG=""
[ "${TRUST_REMOTE_CODE:-false}" = "true" ] && TRUST_FLAG="--trust-remote-code"
PORT=8000
echo "=== SMOKE: $MODEL_ID (tp=$TP dtype=${DTYPE:-auto}) on $(hostname), vllm=$(python -c 'import vllm;print(vllm.__version__)') ==="
t0=$(date +%s)
vllm serve "$MODEL_ID" \
  --dtype "${DTYPE:-auto}" \
  --tensor-parallel-size "$TP" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL:-0.90}" \
  --download-dir "$HF_HOME" \
  $TRUST_FLAG \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port $PORT ${EXTRA_ARGS:-} &
SERVER_PID=$!

ready=0
for i in $(seq 1 300); do
  sleep 10
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && { ready=1; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died at check $i"; break; }
  [ $((i % 6)) -eq 0 ] && echo "  ... waiting ($((i*10))s)"
done

RESULT="FAIL"; LOAD_TIME="--"; TTFT_MS="--"; DECODE_TPS="--"
if [ $ready -eq 1 ]; then
  LOAD_TIME=$(($(date +%s) - t0))
  echo "ready after ${LOAD_TIME}s"
  response=$(curl -s --max-time 600 http://localhost:$PORT/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"'"$SERVED_NAME"'","prompt":"The capital of Austria is","max_tokens":24,"temperature":0.2}' 2>&1) || true
  echo "raw: $(echo "$response" | head -c 400)"
  echo "$response" | grep -q '"choices"' && RESULT="PASS"
  if [ "$RESULT" = "PASS" ]; then
    perf=$(python3 "${GUIDE_DIR:-$HOME/musica-llm-guide}/scripts/measure_perf.py" \
             "http://localhost:$PORT" "$SERVED_NAME" 128 2>&1) || true
    if echo "$perf" | grep -q '^TTFT_MS='; then
      eval "$(echo "$perf" | grep '^TTFT_MS=')"; echo "perf: $perf"
    else
      echo "perf measurement failed (PASS verdict unaffected):"; echo "$perf" | tail -5
    fi
  fi
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
fi
echo "============================================"
echo "RESULT: $RESULT | Load time: ${LOAD_TIME}s | TTFT: ${TTFT_MS}ms | Decode: ${DECODE_TPS} tok/s | Config: $CONFIG"
echo ">>> ${RESULT}: ${MODEL_ID}"
curl -s -H "Title: MUSICA guide: ${RESULT}" -d "${RESULT}: ${SERVED_NAME} (1N smoke, load ${LOAD_TIME}s, TTFT ${TTFT_MS}ms, ${DECODE_TPS} tok/s)" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
kill $SERVER_PID 2>/dev/null; wait 2>/dev/null
[ "$RESULT" = "PASS" ]
