#!/bin/bash
# ============================================================================
# submit_multinode.sh -- Submit a multi-node vLLM job from a config file
# ============================================================================
#
# Usage:
#   scripts/submit_multinode.sh configs/multinode/<model>.conf
#   scripts/submit_multinode.sh configs/multinode/deepseek_v32.conf
#
# Reads NODES and SERVED_NAME from config, submits with correct SLURM args.
# ============================================================================

set -euo pipefail

CONFIG="${1:?Usage: scripts/submit_multinode.sh <config.conf>}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

# Ensure log directory exists
mkdir -p logs/multinode

echo "Submitting: ${SERVED_NAME} (${NODES} nodes, mode=${MODE})"
echo "  Model: ${MODEL_ID}"
echo "  Config: ${CONFIG}"

SBATCH_OUT=$(sbatch \
    -N "${NODES}" \
    --job-name "${SERVED_NAME}" \
    --output "logs/multinode/${SERVED_NAME}_%j.out" \
    --error "logs/multinode/${SERVED_NAME}_%j.err" \
    scripts/run_multinode.sh "$CONFIG" 2>&1)

echo "  $SBATCH_OUT"
JOB_ID=$(echo "$SBATCH_OUT" | grep -oP 'Submitted batch job \K\d+')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "Monitor: tail -f logs/multinode/${SERVED_NAME}_${JOB_ID}.out"
    echo "Errors:  tail -f logs/multinode/${SERVED_NAME}_${JOB_ID}.err"
fi
