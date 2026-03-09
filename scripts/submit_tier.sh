#!/bin/bash
# ============================================================================
# submit_tier.sh -- Submit a single-node tier test
# ============================================================================
#
# Usage:
#   scripts/submit_tier.sh configs/tiers/<tier>.conf
# ============================================================================

set -euo pipefail

CONFIG="${1:?Usage: scripts/submit_tier.sh <tier.conf>}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

mkdir -p logs

echo "Submitting: ${TIER_NAME} tier (${#MODELS[@]} models, ${TIER_GPUS} GPUs)"

SBATCH_OUT=$(sbatch \
    --gres="gpu:${TIER_GPUS}" \
    --job-name "tier_${TIER_NAME}" \
    --output "logs/tier_${TIER_NAME}_%j.out" \
    --error "logs/tier_${TIER_NAME}_%j.err" \
    scripts/run_tier.sh "$CONFIG" 2>&1)

echo "  $SBATCH_OUT"
JOB_ID=$(echo "$SBATCH_OUT" | grep -oP 'Submitted batch job \K\d+')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "Monitor: tail -f logs/tier_${TIER_NAME}_${JOB_ID}.out"
fi
