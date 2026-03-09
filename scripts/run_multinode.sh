#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
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
source "$CONFIG"

# ── Common environment ───────────────────────────────────────
export HF_HOME=/data/fs201045/rl41113/hf-cache
export TRANSFORMERS_CACHE=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export UV_LINK_MODE=copy
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# CUDA_HOME needed for FlashInfer CUTLASS JIT + DeepGEMM FP8 kernel compilation
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

# Allow config to override VENV (e.g. vLLM 0.17.0 for Qwen3.5/GLM-5)
if [ -n "${VENV:-}" ]; then
    VENV="${VENV}/bin/activate"
else
    VENV="/data/fs201045/rl41113/vllm-venv/bin/activate"
fi
source "$VENV"

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
CACHE_USED=$(du -s /data/fs201045/rl41113/hf-cache 2>/dev/null | awk '{print $1}')
CACHE_AVAIL=$(df /data/fs201045/rl41113/ 2>/dev/null | awk 'NR==2{print $4}')
echo "Cache:   $(( ${CACHE_USED:-0} / 1048576 )) GB used, $(( ${CACHE_AVAIL:-0} / 1048576 )) GB free on /data"

if [ "${CACHE_AVAIL:-0}" -lt 524288000 ]; then  # < 500 GB
    echo "WARNING: Less than 500 GB free on /data. Consider cleaning cache."
fi

# ── Pre-download model weights ───────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Pre-downloading model weights..."
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_ID', cache_dir='$HF_HOME'))" 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] Download complete."

# ── Get node topology ────────────────────────────────────────
ALL_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
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
        export HF_HOME=/data/fs201045/rl41113/hf-cache
        echo \"[Ray-head \$(hostname)] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}, nvidia-smi GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)\"
        ray start --block --head --port=${RAY_PORT} --num-gpus=4 --node-ip-address=${HEAD_IP}
      " &
    sleep 15

    # Start Ray workers
    echo "[$(date +%H:%M:%S)] Starting Ray workers"
    for WORKER in $WORKER_NODES; do
        WORKER_IP=$(srun -N 1 -n 1 -w ${WORKER} hostname --ip-address 2>/dev/null | head -1)
        srun -J "ray-worker" -N 1 -n 1 -w ${WORKER} --gpus-per-task=4 \
          bash -c "
            source $VENV
            export HF_HOME=/data/fs201045/rl41113/hf-cache
            echo \"[Ray-worker \$(hostname)] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}, nvidia-smi GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)\"
            ray start --block --address=${HEAD_IP}:${RAY_PORT} --num-gpus=4 --node-ip-address=${WORKER_IP}
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
" 2>/dev/null || echo "  WARNING: Could not verify Ray cluster"

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
        export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
        export PATH=\$CUDA_HOME/bin:\$PATH
        export LIBRARY_PATH=\$CUDA_HOME/targets/x86_64-linux/lib/stubs:\${LIBRARY_PATH:-}"

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
            export HF_HOME=/data/fs201045/rl41113/hf-cache
            export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
            export VLLM_ENGINE_READY_TIMEOUT_S=1800
            ${DEEPGEMM_EXPORTS}
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
                --download-dir /data/fs201045/rl41113/hf-cache \
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
        export HF_HOME=/data/fs201045/rl41113/hf-cache
        export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
        export VLLM_ENGINE_READY_TIMEOUT_S=1800
        ${DEEPGEMM_EXPORTS}
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
            --download-dir /data/fs201045/rl41113/hf-cache \
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
for i in $(seq 1 360); do
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
echo "  Config:    $CONFIG"
echo "============================================================"
echo ">>> ${RESULT}: ${MODEL_ID}"
echo "Finished at $(date)"

# ── Send notification ────────────────────────────────────────
MSG="${RESULT}: ${SERVED_NAME} (${NODES}N ${MODE})"
if [ "$RESULT" = "PASS" ]; then
    MSG="${MSG} - loaded in ${LOAD_TIME}s"
fi
curl -s -d "$MSG" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true

# ── Cleanup ──────────────────────────────────────────────────
if [ "$MODE" = "pp" ]; then
    ray stop 2>/dev/null || true
fi
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true
