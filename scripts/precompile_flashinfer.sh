#!/bin/bash
# ============================================================================
# precompile_flashinfer.sh -- Pre-compile FlashInfer CUTLASS MoE kernels
# ============================================================================
#
# WHY: FlashInfer JIT-compiles ~182 CUTLASS .o files on first use. In DP+EP mode
# 12-24 engine cores all race for the ninja lock on shared NFS $HOME; the build
# collapses and every gated job dies. Run this ONCE per (flashinfer version, venv)
# on a login node — CPU only, no GPU — to produce the cached .so.
#
# Usage:
#   scripts/precompile_flashinfer.sh [VENV] [VERSION]
#     VENV     default /data/fs201045/rl41113/vllm-nightly-venv
#     VERSION  default: auto-detected from that venv's flashinfer
#
# Runtime: ~1-2h on a login node. It detaches itself, so it survives your
# session closing — that is how the 2026-08-28 attempt died (reaped at step
# 3/183 while its status file still read RUNNING, stalling two jobs for 5 days).
#
# VERIFY BY THE .so EXISTING. Never by exit code, never by the status file.
# ============================================================================

set -uo pipefail

VENV="${1:-/data/fs201045/rl41113/vllm-nightly-venv}"
CUDA_HOME="/data/fs201045/rl41113/cuda-nvcc-env"   # nvcc 12.9 — 13.1 breaks CUTLASS JIT

[ -d "$CUDA_HOME" ] || { echo "ERROR: CUDA_HOME missing: $CUDA_HOME"; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv missing: $VENV"; exit 1; }

source "$VENV/bin/activate"
VERSION="${2:-$(python -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null)}"
[ -n "$VERSION" ] || { echo "ERROR: could not detect flashinfer version in $VENV"; exit 1; }

BUILD_DIR="$HOME/.cache/flashinfer/$VERSION/90a/cached_ops/fused_moe_90"
SO="$BUILD_DIR/fused_moe_90.so"
STATUS="/data/fs201045/rl41113/fi_moe90_build.status"
LOG="/data/fs201045/rl41113/fi_moe90_build.log"

echo "=== FlashInfer fused_moe_90 precompile ==="
echo "venv=$VENV  flashinfer=$VERSION"
echo "build dir=$BUILD_DIR"

# Already done? The .so is the only authority.
if [ -f "$SO" ]; then
    echo "ALREADY COMPILED:"; ls -lh "$SO"; echo OK > "$STATUS"; exit 0
fi

# build.ninja is emitted by flashinfer on first import of the op.
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "ERROR: no build.ninja in $BUILD_DIR — run one vLLM MoE job first to generate it."
    exit 1
fi

# Refuse to race a live build (the whole point of this script).
if pgrep -f "ninja -C $BUILD_DIR" >/dev/null; then
    echo "ERROR: a ninja build for this dir is already running:"
    pgrep -af "ninja -C $BUILD_DIR"
    exit 1
fi

# A zero-byte .ninja_lock left by a killed build blocks every future run.
[ -f "$BUILD_DIR/.ninja_lock" ] && { echo "removing stale .ninja_lock"; rm -f "$BUILD_DIR/.ninja_lock"; }

# Detach so a closing session cannot reap the build.
if [ "${FI_DETACHED:-0}" != "1" ]; then
    echo "detaching (log: $LOG)"
    FI_DETACHED=1 setsid nohup "$0" "$VENV" "$VERSION" >"$LOG" 2>&1 < /dev/null &
    sleep 2
    echo "started pid $(pgrep -f "ninja -C $BUILD_DIR" || echo 'pending')  — watch: tail -f $LOG"
    exit 0
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
# The final link step needs the libcuda stub; without these it fails after ~2h of .o work.
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo RUNNING > "$STATUS"
echo "START $(date) host=$(hostname) nvcc=$(nvcc --version 2>&1 | tail -1)"
echo "existing .o: $(ls "$BUILD_DIR"/*.o 2>/dev/null | wc -l) / 182"

nice -n 15 ninja -C "$BUILD_DIR" -j4
rc=$?
echo "ninja rc=$rc $(date)  .o=$(ls "$BUILD_DIR"/*.o 2>/dev/null | wc -l)"

# ── The .so is the verdict. rc is advisory. ──────────────────────────────────
if [ -f "$SO" ]; then
    chmod a-w "$BUILD_DIR"/*.o "$SO" 2>/dev/null || true
    echo OK > "$STATUS"
    echo "BUILD-OK $(date)"; ls -lh "$SO"
    curl -s -H "Title: MUSICA guide" -d "fused_moe_90 precompiled OK (flashinfer $VERSION, $(basename "$VENV")). Gated jobs can be submitted." ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
else
    echo FAIL > "$STATUS"
    echo "BUILD-FAIL $(date) — no .so produced (rc=$rc)"
    curl -s -H "Title: MUSICA guide" -d "fused_moe_90 precompile FAILED (flashinfer $VERSION, rc=$rc, no .so). Next: compute-node job via scripts/compile_flashinfer_slurm.sh." ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
    exit 1
fi
