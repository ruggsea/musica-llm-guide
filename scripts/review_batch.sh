#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH -A p201276
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time=04:00:00

# Batch-score chat prompts with one single-node model: first-token top-20 logprobs per line (scripts/review_client.py).
# Serves the model exactly as its configs/single2026/<model>.conf passed, with a longer context for real pages.
#   sbatch -J review-mimo -o logs/review/%x_%j.out scripts/review_batch.sh configs/single2026/mimo_v25.conf <in.jsonl> <out.jsonl>
# Env: REVIEW_MAX_MODEL_LEN (default 32768), REVIEW_MAX_NUM_SEQS (64), REVIEW_CONCURRENCY (64),
#      REVIEW_TEMPLATE_KWARGS (JSON, default '{"enable_thinking": false}'),
#      REVIEW_SAMPLE_N / REVIEW_SAMPLE_TOKENS (default 0 / 30): also write N prompts with 30 tokens of text to <out>.sample30.jsonl
# Resumable: rerun with the same output file and only unscored ids are sent.
set -uo pipefail
CONFIG="${1:?usage: review_batch.sh <conf> <in.jsonl> <out.jsonl>}"; IN="${2:?}"; OUT="${3:?}"
source ~/musica-env.sh
source "$CONFIG"
source "${VENV:-$MUSICA_VENV}/bin/activate"
[ -n "${PY_OVERLAY:-}" ] && export PYTHONPATH=$PY_OVERLAY:${PYTHONPATH:-} && echo "PY_OVERLAY: $PY_OVERLAY"
export REVIEW_TEMPLATE_KWARGS="${REVIEW_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}"
GUIDE=${GUIDE_DIR:-$HOME/musica-llm-guide}

# Compiled-kernel check: rebuild a stale FlashInfer kernel once here, before the TP ranks start and race on it
FI_VER=$(python -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null)
STALE=$([ -n "$FI_VER" ] && ~/musica-setup/fi-cache-check.sh 2>&1 | grep "^STALE" | grep " ${FI_VER}/" | grep -v "\.stale")
for d in $(echo "$STALE" | awk '{print $2}'); do
    t0=$(date +%s); ~/musica-setup/fi-cache-check.sh --fix "$d" | grep -v -E "^(ok|STALE) "
    echo "Kernels: rebuilt $d once in $(( $(date +%s) - t0 ))s"
done

TRUST_FLAG=""; [ "${TRUST_REMOTE_CODE:-false}" = "true" ] && TRUST_FLAG="--trust-remote-code"
PORT=8000
echo "=== REVIEW: $MODEL_ID tp=$TP on $(hostname), vllm=$(python -c 'import vllm;print(vllm.__version__)'), in=$IN out=$OUT ==="
echo "max_model_len=${REVIEW_MAX_MODEL_LEN:-32768} max_num_seqs=${REVIEW_MAX_NUM_SEQS:-64} concurrency=${REVIEW_CONCURRENCY:-64} template_kwargs=$REVIEW_TEMPLATE_KWARGS"
t0=$(date +%s)
vllm serve "$MODEL_ID" \
  --dtype "${DTYPE:-auto}" \
  --tensor-parallel-size "$TP" \
  --max-model-len "${REVIEW_MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${REVIEW_MAX_NUM_SEQS:-64}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL:-0.90}" \
  --max-logprobs 20 \
  --download-dir "$HF_HOME" \
  $TRUST_FLAG \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port $PORT ${EXTRA_ARGS:-} &
SERVER_PID=$!

ready=0
for i in $(seq 1 360); do
  sleep 10
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && { ready=1; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died at check $i"; break; }
  [ $((i % 6)) -eq 0 ] && echo "  ... waiting ($((i*10))s)"
done
[ $ready -eq 1 ] || { echo "REVIEW FAIL: server not ready"; kill $SERVER_PID 2>/dev/null; exit 2; }
echo "ready after $(( $(date +%s) - t0 ))s"
python3 "$GUIDE/scripts/sanity_check.py" "http://localhost:$PORT" "$SERVED_NAME" || { echo "REVIEW FAIL: model answers are not sane, nothing scored"; kill $SERVER_PID 2>/dev/null; exit 3; }

python3 "$GUIDE/scripts/review_client.py" "http://localhost:$PORT" "$SERVED_NAME" "$IN" "$OUT" "${REVIEW_CONCURRENCY:-64}" 20
rc=$?
echo "REVIEW client exit=$rc, output lines: $(wc -l < "$OUT")"
# Readable sample: first N prompts with REVIEW_SAMPLE_TOKENS of text, to see what the answer really starts with
if [ "${REVIEW_SAMPLE_N:-0}" -gt 0 ]; then
  REVIEW_MAX_TOKENS=${REVIEW_SAMPLE_TOKENS:-30} REVIEW_LIMIT=$REVIEW_SAMPLE_N \
    python3 "$GUIDE/scripts/review_client.py" "http://localhost:$PORT" "$SERVED_NAME" "$IN" "${OUT%.jsonl}.sample${REVIEW_SAMPLE_TOKENS:-30}.jsonl" 4 20
fi
kill $SERVER_PID 2>/dev/null; wait 2>/dev/null
exit $rc
