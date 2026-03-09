#!/bin/bash
#SBATCH --job-name "flashinfer"
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=02:00:00
#SBATCH --output=logs/multinode/flashinfer_compile_%j.out
#SBATCH --error=logs/multinode/flashinfer_compile_%j.err

# Pre-compile FlashInfer CUTLASS MoE kernels on a compute node.
# Login nodes kill long-running CPU processes. Compute nodes don't.
# Only needs CPU (nvcc), but must request GPUs for this partition.

set -eo pipefail

BUILD_DIR="$HOME/.cache/flashinfer/0.6.1/90a/cached_ops/fused_moe_90"
VENV="/data/fs201045/rl41113/vllm-venv/bin/activate"
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

source "$VENV"

# Fix: linker needs libcuda.so and libnvrtc.so which are driver libs, not in micromamba CUDA env.
# The build.ninja hardcodes -L paths to $CUDA_HOME/lib64/stubs, so we symlink there.
STUBS_DIR="$CUDA_HOME/lib64/stubs"
mkdir -p "$STUBS_DIR"
for lib in libcuda.so libnvrtc.so; do
    if [ ! -f "$STUBS_DIR/$lib" ]; then
        SYSTEM_LIB=$(find /usr/lib64 /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu -name "${lib}*" \( -type f -o -type l \) 2>/dev/null | head -1 || true)
        if [ -n "${SYSTEM_LIB:-}" ]; then
            ln -sf "$SYSTEM_LIB" "$STUBS_DIR/$lib"
            echo "Symlinked $lib -> $SYSTEM_LIB"
        else
            echo "WARNING: Could not find $lib on system"
        fi
    fi
done

echo "=== FlashInfer CUTLASS MoE Kernel Compilation ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "nvcc: $(nvcc --version 2>&1 | tail -1)"

# Unlock compiled artifacts if previously chmod'd read-only
for f in "$BUILD_DIR"/*.o "$BUILD_DIR"/*.so; do
    [ -f "$f" ] && [ ! -w "$f" ] && chmod u+w "$f"
done
echo "Unlocked compiled artifacts for recompilation"

# Remove stale lock and failed .so (so ninja retries the link step)
rm -f "$BUILD_DIR/.ninja_lock"
rm -f "$BUILD_DIR/fused_moe_90.so"

O_COUNT=$(find "$BUILD_DIR" -maxdepth 1 -name '*.o' 2>/dev/null | wc -l)
echo "Existing .o files: $O_COUNT / 182"

echo ""
echo "Starting ninja -j32 (compute node has 192 cores)..."
/data/fs201045/rl41113/vllm-venv/bin/ninja -C "$BUILD_DIR" -j32

echo ""
echo "=== Compilation complete ==="
echo "Finished: $(date)"

if ls "$BUILD_DIR"/*.so 2>/dev/null; then
    echo "SUCCESS!"
    ls -lh "$BUILD_DIR"/*.so
    # CRITICAL: Make compiled artifacts read-only to prevent NFS rebuild race.
    # Workers on different nodes trigger ninja to check .o freshness;
    # NFS timestamp differences cause false "stale" detection and 60+ min rebuilds.
    # Only protect .o and .so files — ninja metadata (.ninja_deps, .ninja_log)
    # must remain writable or ninja fails with "Permission denied".
    chmod a-w "$BUILD_DIR"/*.o "$BUILD_DIR"/*.so
    echo "Compiled artifacts (.o, .so) set to READ-ONLY (prevents NFS rebuild race)"
    curl -s -d "FlashInfer CUTLASS kernels compiled + locked! Ready for V3.2 + Mistral" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
else
    echo "FAILED - no .so produced"
    curl -s -d "FlashInfer compilation FAILED on compute node" ntfy.sh/ruggsea-vsc >/dev/null 2>&1 || true
    exit 1
fi
