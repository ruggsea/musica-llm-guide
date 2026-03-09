#!/bin/bash
# ============================================================================
# precompile_flashinfer.sh -- Pre-compile FlashInfer CUTLASS MoE kernels
# ============================================================================
#
# WHY: FlashInfer JIT-compiles ~182 CUTLASS .o files on first use. With DP+EP
# mode running 12+ engine cores, they all compete for the ninja lock on shared
# NFS home, taking 30+ minutes (exceeding all timeouts).
#
# FIX: Run this ONCE on the login node (CPU-only, no GPU needed). It produces
# a cached .so that all future jobs load instantly.
#
# Requirements:
#   - nvcc 12.9 (NOT 13.1 -- CCCL headers break CUTLASS JIT)
#   - Source the vLLM venv first
#
# Runtime: ~60-90 minutes on login node (CPU-bound compilation)
# ============================================================================

set -euo pipefail

VENV="/data/fs201045/rl41113/vllm-venv/bin/activate"
CUDA_HOME="/data/fs201045/rl41113/cuda-nvcc-env"
BUILD_DIR="$HOME/.cache/flashinfer/0.6.1/90a/cached_ops/fused_moe_90"
NINJA="/data/fs201045/rl41113/vllm-venv/bin/ninja"

echo "=== FlashInfer CUTLASS MoE Kernel Pre-compilation ==="
echo "Build dir: $BUILD_DIR"
echo "Started: $(date)"

# Check prerequisites
if [ ! -d "$CUDA_HOME" ]; then
    echo "ERROR: CUDA_HOME not found: $CUDA_HOME"
    echo "Install: micromamba install -p cuda-nvcc-env -c nvidia cuda-nvcc=12.9 cuda-cuobjdump=12.9"
    exit 1
fi

if [ ! -f "$NINJA" ]; then
    echo "ERROR: ninja not found at $NINJA"
    exit 1
fi

# Check if already compiled
if ls "$BUILD_DIR"/*.so 2>/dev/null; then
    echo "Already compiled! .so file exists:"
    ls -lh "$BUILD_DIR"/*.so
    exit 0
fi

# Check if build.ninja exists (created by FlashInfer on first import)
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "ERROR: build.ninja not found. Run a vLLM MoE model once to generate it."
    echo "Expected: $BUILD_DIR/build.ninja"
    exit 1
fi

# Remove stale lock if present
if [ -f "$BUILD_DIR/.ninja_lock" ]; then
    echo "Removing stale .ninja_lock"
    rm -f "$BUILD_DIR/.ninja_lock"
fi

# Setup environment
source "$VENV"
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

echo "nvcc: $(nvcc --version 2>&1 | tail -1)"
echo ""

# Count existing .o files
EXISTING=$(ls "$BUILD_DIR"/*.o 2>/dev/null | wc -l)
echo "Existing .o files: $EXISTING / 182"
echo ""

# Run compilation
echo "Starting ninja -j4 ..."
"$NINJA" -C "$BUILD_DIR" -j4

echo ""
echo "=== Compilation complete ==="
echo "Finished: $(date)"

if ls "$BUILD_DIR"/*.so 2>/dev/null; then
    echo "SUCCESS: .so file created"
    ls -lh "$BUILD_DIR"/*.so
    curl -s -d "FlashInfer CUTLASS kernels compiled successfully" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
else
    echo "ERROR: No .so file found after compilation"
    curl -s -d "FlashInfer compilation failed - no .so produced" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
    exit 1
fi
