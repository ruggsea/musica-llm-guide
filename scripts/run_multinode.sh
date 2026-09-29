#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

# ============================================================================
# run_multinode.sh -- Universal multi-node vLLM runner
# ============================================================================
#
# Handles both Ray+PP and DP+EP modes based on config.
#
# Usage:
#   sbatch -N <nodes> --job-name <name> \
#     -o logs/multinode/<name>_%j.out -e logs/multinode/<name>_%j.err \
#     scripts/run_multinode.sh configs/multinode/<model>.conf
#
# Or use the submit helper:
#   scripts/submit_multinode.sh configs/multinode/<model>.conf
#
# Modes:
#   pp   -- Ray cluster + vllm serve --distributed-executor-backend ray
#   dpep -- headless workers + master (no Ray), --enable-expert-parallel
#
# Prerequisites for DP+EP MoE models:
#   - Pre-compile FlashInfer CUTLASS kernels: scripts/precompile_flashinfer.sh
#   - DeepGEMM needs nvcc 12.9 + cuobjdump (CUDA_HOME set automatically)
# ============================================================================

set -uo pipefail

# ── Load config ──────────────────────────────────────────────
CONFIG="${1:?Usage: scripts/run_multinode.sh <config.conf>}"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi
# Standard env, identical at vie/inn/lnz (sets MUSICA_ROOT, HF_HOME, VLLM_CACHE_ROOT, CUDA_HOME, MAX_JOBS...).
# Sourced BEFORE the config so a config can say VENV=$MUSICA_ROOT/venvs/vllm-nightly.
source ~/musica-env.sh
source "$CONFIG"

# --- abnormal-termination notifier (preemption, node failure, walltime kill) ---
RESULT="${RESULT:-}"
_notify_abnormal(){ if [ -z "${RESULT:-}" ] || [ "${RESULT:-}" = "" ]; then
  curl -s -H "Title: MUSICA guide: ABORTED" -d "ABORTED (no result): ${SERVED_NAME:-job} slurm=${SLURM_JOB_ID:-?} — preempted/killed/node-fail" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
fi; }
trap _notify_abnormal EXIT


# ── Common environment (on top of ~/musica-env.sh; srun steps inherit it) ──
export TRANSFORMERS_CACHE=$HF_HOME
export UV_LINK_MODE=copy
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# nvcc + libcuda stubs for FlashInfer CUTLASS JIT + DeepGEMM FP8 kernel compilation
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Config may pick another standard venv: $MUSICA_ROOT/venvs/{vllm-nightly,vllm-0.17}
VENV="${VENV:-$MUSICA_VENV}/bin/activate"
source "$VENV"
echo "Site:    ${MUSICA_SITE}  venv: ${VENV%/bin/activate}  HF_HOME: $HF_HOME"

# ── Compiled-kernel check (compiles nothing) ─────────────────
# A FlashInfer kernel that ninja sees as stale gets rebuilt by EVERY rank at once in the same dir
# (the Aug 2026 fused_moe_90 race). Abort instead of letting the ranks race.
FI_VER=$(python -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null)
STALE=$(~/musica-setup/fi-cache-check.sh 2>&1 | grep "^STALE" | grep " ${FI_VER}/" | grep -v "\.stale")
if [ -n "$STALE" ]; then
    # Rebuild once here, single process, before any rank starts; otherwise every rank compiles it at once.
    # Limit: this builds the recipe on disk. fused_moe_90's source list differs per model (FlashInfer 0.6.17),
    # so if another model wrote the recipe last, the ranks can still regenerate and rebuild it.
    echo "Kernels: FlashInfer ${FI_VER} stale, rebuilding once on $(hostname) before starting ranks:"
    echo "$STALE"
    for d in $(echo "$STALE" | awk '{print $2}'); do
        t0=$(date +%s)
        MAX_JOBS=8 ~/musica-setup/fi-cache-check.sh --fix "$d" | grep -v -E "^(ok|STALE) "
        echo "  rebuilt $d in $(( $(date +%s) - t0 ))s"
    done
    STALE=$(~/musica-setup/fi-cache-check.sh 2>&1 | grep "^STALE" | grep " ${FI_VER}/" | grep -v "\.stale")
    if [ -n "$STALE" ]; then
        echo "ABORT: still stale after rebuild:"; echo "$STALE"; exit 3
    fi
fi
echo "Kernels: FlashInfer ${FI_VER} cache ok"

PORT=8000
RAY_PORT=6379
RPC_PORT=29600
# Avoid EADDRINUSE on torch.distributed init port
export MASTER_PORT=45200

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# ── Startup info ─────────────────────────────────────────────
echo "============================================================"
echo "Multi-node vLLM: ${SERVED_NAME}"
echo "============================================================"
echo "Config:  $CONFIG"
echo "Model:   $MODEL_ID"
echo "Mode:    $MODE (nodes=$NODES, tp=$TP)"
echo "Started: $(date)"
echo "Nodes:   $SLURM_NNODES ($SLURM_JOB_NODELIST)"

echo "nvcc:    $(nvcc --version 2>&1 | tail -1)"
if [ "${NEEDS_DEEPGEMM:-false}" = "true" ]; then
    echo "cuobjdump: $(cuobjdump --version 2>&1 | head -1)"
fi

# ── Cache space check ────────────────────────────────────────
# Quota, not df: df shows the whole filesystem at inn/lnz. See $MUSICA_ROOT/CACHE_COORDINATION.md.
/usr/lpp/mmfs/bin/mmlsquota -j fs201045 $(df --output=source /data | tail -1) 2>/dev/null | tail -1 \
  | awk '{printf "Cache:   %.0f GB used of %.0f GB quota on fs201045\n", $3/1e6*1.024, $4/1e6*1.024}'

# ── Pre-download model weights ───────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Pre-downloading model weights..."
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_ID', cache_dir='$HF_HOME'))" 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] Download complete."

# ── Get node topology ────────────────────────────────────────
# scontrol can come back empty on a transient controller hiccup (job 1773023: no head node, Ray never started,
# vLLM waited 40+ min for a 16-GPU placement group). Retry, then abort instead of hanging.
for try in 1 2 3; do ALL_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST); [ -n "$ALL_NODES" ] && break; sleep 10; done
[ -n "$ALL_NODES" ] || { echo "ABORT: scontrol show hostnames  returned nothing"; exit 4; }
HEAD_NODE=$(echo "$ALL_NODES" | head -n 1)
HEAD_IP=$(srun -N 1 -n 1 -w ${HEAD_NODE} hostname --ip-address 2>/dev/null | head -1)
WORKER_NODES=$(echo "$ALL_NODES" | tail -n +2)

echo ""
echo "Head: ${HEAD_NODE} (${HEAD_IP})"
echo "Workers: $(echo $WORKER_NODES | tr '\n' ' ')"

t0=$(date +%s)

# ── Build trust-remote-code flag ─────────────────────────────
TRUST_FLAG=""
if [ "${TRUST_REMOTE_CODE:-false}" = "true" ]; then
    TRUST_FLAG="--trust-remote-code"
fi

# ============================================================
# MODE: PP (Ray + Pipeline Parallel)
# ============================================================
if [ "$MODE" = "pp" ]; then
    echo ""
    echo "--- Ray + Pipeline Parallel (TP=$TP, PP=$PP) ---"

    # Start Ray head
    echo "[$(date +%H:%M:%S)] Starting Ray head on ${HEAD_NODE}"
    srun -J "ray-head" -N 1 -n 1 -w ${HEAD_NODE} --gpus-per-task=4 \
      bash -c "
        source $VENV
        echo \"[Ray-head \$(hostname)] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}, nvidia-smi GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)\"
        ulimit -n 65536; ray start --block --head --port=${RAY_PORT} --num-gpus=4 --node-ip-address=${HEAD_IP}
      " &
    sleep 15

    # Start Ray workers
    echo "[$(date +%H:%M:%S)] Starting Ray workers"
    for WORKER in $WORKER_NODES; do
        # getent needs no job step; an srun here came back empty once when a node's prolog hung (1776673)
        WORKER_IP=$(getent hosts ${WORKER} | awk '{print $1; exit}')
        [ -z "$WORKER_IP" ] && WORKER_IP=$(srun -N 1 -n 1 -w ${WORKER} hostname --ip-address 2>/dev/null | head -1)
        [ -z "$WORKER_IP" ] && { echo "RESULT: FAIL (no IP for worker ${WORKER})"; exit 1; }
        srun -J "ray-worker" -N 1 -n 1 -w ${WORKER} --gpus-per-task=4 \
          bash -c "
            source $VENV
            echo \"[Ray-worker \$(hostname)] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}, nvidia-smi GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)\"
            ulimit -n 65536; ray start --block --address=${HEAD_IP}:${RAY_PORT} --num-gpus=4 --node-ip-address=${WORKER_IP}
          " &
    done
    sleep 25

    # Verify Ray cluster
    echo "[$(date +%H:%M:%S)] Verifying Ray cluster..."
    python3 -c "
import ray; ray.init(address='${HEAD_IP}:${RAY_PORT}')
r = ray.cluster_resources(); n = [x for x in ray.nodes() if x['Alive']]
print(f'  GPUs: {r.get(\"GPU\",0)}, Nodes: {len(n)}')
ray.shutdown()
raise SystemExit(0 if r.get(\"GPU\",0) >= ${TP}*${PP} else 3)
" 2>/dev/null
    RAY_OK=$?
    # vLLM would otherwise wait 30 min for GPUs that never join (1776673: a worker failed, 12 of 16 GPUs)
    [ $RAY_OK -eq 3 ] && { echo "RESULT: FAIL (Ray cluster has fewer than $((TP*PP)) GPUs)"; exit 1; }
    [ $RAY_OK -ne 0 ] && echo "  WARNING: Could not verify Ray cluster"

    # Launch vLLM serve
    echo "[$(date +%H:%M:%S)] Launching vLLM serve (PP mode)"
    vllm serve "$MODEL_ID" \
        --dtype "${DTYPE:-auto}" \
        --tensor-parallel-size "$TP" \
        --pipeline-parallel-size "$PP" \
        --distributed-executor-backend ray \
        --download-dir "$HF_HOME" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "${MAX_NUM_SEQS:-16}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        $TRUST_FLAG \
        --host 0.0.0.0 \
        --port $PORT \
        --served-model-name "$SERVED_NAME" \
        ${EXTRA_ARGS:-} &
    SERVER_PID=$!

# ============================================================
# MODE: DP+EP (Data Parallel + Expert Parallel)
# ============================================================
elif [ "$MODE" = "dpep" ]; then
    echo ""
    echo "--- DP+EP (dp_size=$DP_SIZE, dp_local=$DP_LOCAL) ---"

    # Common env exports for srun workers (CUDA_HOME always needed for FlashInfer)
    DEEPGEMM_EXPORTS="
        export CUDA_HOME=$CUDA_HOME
        export PATH=\$CUDA_HOME/bin:\$PATH
        export LIBRARY_PATH=\$CUDA_HOME/targets/x86_64-linux/lib/stubs:\${LIBRARY_PATH:-}
        export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"

    # Launch headless workers
    RANK=0
    for NODE in $ALL_NODES; do
        if [ "$NODE" = "$HEAD_NODE" ]; then
            RANK=$((RANK + DP_LOCAL))
            continue
        fi
        echo "[$(date +%H:%M:%S)] Worker on ${NODE} (start_rank=$RANK)"
        srun -N 1 -n 1 -w ${NODE} --gpus-per-task=4 \
          bash -c "
            source $VENV
            export VLLM_ENGINE_READY_TIMEOUT_S=1800
            ${DEEPGEMM_EXPORTS}
            ${EXTRA_ENV_EXPORTS:-}
            echo \"[Worker \$(hostname)] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}, nvidia-smi GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)\"
            vllm serve $MODEL_ID \
                --headless \
                --data-parallel-start-rank $RANK \
                $TRUST_FLAG \
                --data-parallel-size $DP_SIZE \
                --data-parallel-size-local $DP_LOCAL \
                --data-parallel-address ${HEAD_IP} \
                --data-parallel-rpc-port $RPC_PORT \
                --enable-expert-parallel \
                --download-dir $HF_HOME \
                --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-4096} \
                --max-num-seqs ${MAX_NUM_SEQS:-16} \
                --max-model-len $MAX_MODEL_LEN \
                --gpu-memory-utilization $GPU_MEMORY_UTIL \
                ${EXTRA_ARGS:-}
          " &
        RANK=$((RANK + DP_LOCAL))
    done
    sleep 10

    # Launch master
    echo "[$(date +%H:%M:%S)] Master on ${HEAD_NODE}"
    srun -N 1 -n 1 -w ${HEAD_NODE} --gpus-per-task=4 \
      bash -c "
        source $VENV
        export VLLM_ENGINE_READY_TIMEOUT_S=1800
        ${DEEPGEMM_EXPORTS}
        ${EXTRA_ENV_EXPORTS:-}
        echo \"[Master \$(hostname)] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}, nvidia-smi GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)\"
        vllm serve $MODEL_ID \
            --port $PORT \
            --served-model-name $SERVED_NAME \
            $TRUST_FLAG \
            --data-parallel-size $DP_SIZE \
            --data-parallel-size-local $DP_LOCAL \
            --data-parallel-address ${HEAD_IP} \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-expert-parallel \
            --download-dir $HF_HOME \
            --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-4096} \
            --max-num-seqs ${MAX_NUM_SEQS:-16} \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEMORY_UTIL \
            --host 0.0.0.0 \
            ${EXTRA_ARGS:-}
      " &
    SERVER_PID=$!

else
    echo "ERROR: Unknown mode '$MODE'. Use 'pp' or 'dpep'."
    exit 1
fi

# ============================================================
# Health check + generation test
# ============================================================
echo ""
echo "[$(date +%H:%M:%S)] Waiting for server (PID $SERVER_PID)..."

ready=0
for i in $(seq 1 ${WAIT_CHECKS:-360}); do
    sleep 10
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
        ready=1
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  Server process died at check $i"
        break
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "  ... waiting ($((i*10))s / 3600s)"
    fi
done

RESULT="FAIL"
LOAD_TIME="--"
TTFT_MS="--"
DECODE_TPS="--"

if [ $ready -eq 1 ]; then
    LOAD_TIME=$(($(date +%s) - t0))
    echo "[$(date +%H:%M:%S)] Server ready after ${LOAD_TIME}s"
    echo ""
    echo "Testing generation (10min timeout for first request)..."
    response=$(curl -s --max-time 600 http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"'"$SERVED_NAME"'","prompt":"Hello, my name is","max_tokens":32,"temperature":0.7}' 2>&1) || true

    if echo "$response" | grep -q "choices"; then
        RESULT="PASS"
        output=$(echo "$response" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:200])' 2>/dev/null || echo "$response" | head -c 300)
        echo "[$(date +%H:%M:%S)] Generation successful!"
        echo "  Output: $output"
        # Coherence check: a reply coming back is not a pass (MiMo-V2.5 "passed" 2026-09-27 on gibberish)
        python3 "${GUIDE_DIR:-$HOME/musica-llm-guide}/scripts/sanity_check.py" "http://localhost:${PORT}" "$SERVED_NAME" || { RESULT="FAIL"; echo "FAIL: answers are not sane (see SANITY lines)"; }
        echo "[$(date +%H:%M:%S)] Measuring single-stream perf (TTFT + decode tok/s)..."
        perf=$(python3 "${GUIDE_DIR:-$HOME/musica-llm-guide}/scripts/measure_perf.py" \
                 "http://localhost:${PORT}" "$SERVED_NAME" 128 2>&1) || true
        if echo "$perf" | grep -q '^TTFT_MS='; then
            eval "$(echo "$perf" | grep '^TTFT_MS=')"
            echo "  $perf"
        else
            echo "  perf measurement failed (PASS verdict unaffected):"
            echo "$perf" | tail -5
        fi
        # Optional work against the live server once it passed (e.g. a batched review): conf sets POST_PASS_CMD
        if [ "$RESULT" = "PASS" ] && [ -n "${POST_PASS_CMD:-}" ]; then
            echo "[$(date +%H:%M:%S)] POST_PASS_CMD: $POST_PASS_CMD"
            BASE_URL="http://localhost:${PORT}" SERVED_NAME="$SERVED_NAME" bash -c "$POST_PASS_CMD"
            echo "[$(date +%H:%M:%S)] POST_PASS_CMD exit $?"
        fi
    else
        echo "Generation failed: $(echo "$response" | head -c 500)"
    fi
else
    echo "Server failed to start within 3600s"
fi

# ============================================================
# Report results
# ============================================================
echo ""
echo "============================================================"
echo "RESULT: $RESULT"
echo "  Model:     $MODEL_ID"
echo "  Mode:      $MODE (nodes=$NODES)"
echo "  Load time: ${LOAD_TIME}s"
echo "  TTFT:      ${TTFT_MS}ms (single stream)"
echo "  Decode:    ${DECODE_TPS} tok/s (single stream)"
echo "  Config:    $CONFIG"
echo "============================================================"
echo ">>> ${RESULT}: ${MODEL_ID}"
echo "Finished at $(date)"

# ── Send notification ────────────────────────────────────────
MSG="${RESULT}: ${SERVED_NAME} (${NODES}N ${MODE})"
if [ "$RESULT" = "PASS" ]; then
    MSG="${MSG} - loaded in ${LOAD_TIME}s, TTFT ${TTFT_MS}ms, ${DECODE_TPS} tok/s"
fi
curl -s -H "Title: MUSICA guide: ${RESULT}" -d "$MSG" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true

# ── Cleanup or stay alive ─────────────────────────────────────
if [ "${KEEP_ALIVE:-0}" = "1" ] && [ "$RESULT" = "PASS" ]; then
    echo "KEEP_ALIVE=1 — staying alive for inference until SLURM timeout or external scancel"
    # Wait on the server PID — exits when vLLM dies or job is cancelled
    wait $SERVER_PID 2>/dev/null
    echo "Server exited."
fi
if [ "$MODE" = "pp" ]; then
    ray stop 2>/dev/null || true
fi
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true
