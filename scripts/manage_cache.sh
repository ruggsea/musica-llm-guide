#!/bin/bash
# ============================================================================
# manage_cache.sh -- HF model cache management
# ============================================================================
#
# Usage:
#   scripts/manage_cache.sh status          # Show cache usage
#   scripts/manage_cache.sh list            # List cached models
#   scripts/manage_cache.sh cleanup         # Interactive cleanup prompts
#   scripts/manage_cache.sh delete <repo>   # Delete a specific model
#
# Cache location: /data/fs201045/rl41113/hf-cache
# ============================================================================

set -euo pipefail

HF_CACHE="/data/fs201045/rl41113/hf-cache"
VLLM_CACHE="/data/fs201045/rl41113/vllm-cache"

case "${1:-status}" in
    status)
        echo "=== Cache Status ==="
        echo ""
        echo "HF Cache: $HF_CACHE"
        du -sh "$HF_CACHE" 2>/dev/null || echo "  Not found"
        echo ""
        echo "vLLM Cache: $VLLM_CACHE"
        du -sh "$VLLM_CACHE" 2>/dev/null || echo "  Not found"
        echo ""
        echo "Home FlashInfer: ~/.cache/flashinfer/"
        du -sh ~/.cache/flashinfer/ 2>/dev/null || echo "  Not found"
        echo ""
        echo "/data partition:"
        df -h /data/fs201045/rl41113/ 2>/dev/null
        echo ""
        echo "Home partition:"
        df -h ~ 2>/dev/null
        ;;

    list)
        echo "=== Cached Models ==="
        echo ""
        if [ -d "$HF_CACHE/models--*" ] 2>/dev/null || [ -d "$HF_CACHE/hub/models--*" ] 2>/dev/null; then
            # Check both old and new HF cache layouts
            for dir in "$HF_CACHE"/models--* "$HF_CACHE"/hub/models--* 2>/dev/null; do
                [ -d "$dir" ] || continue
                model_name=$(basename "$dir" | sed 's/models--//' | sed 's/--/\//g')
                size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
                echo "  $size  $model_name"
            done | sort -rh
        else
            echo "  No models found in cache"
        fi
        ;;

    delete)
        REPO="${2:?Usage: manage_cache.sh delete <org/model>}"
        CACHE_NAME=$(echo "$REPO" | sed 's/\//--%2F--/g; s/\//--%2F/g; s/\//--%2F/g' | sed 's/\//--%2F/g')
        # Try both naming conventions
        SAFE_NAME=$(echo "$REPO" | sed 's/\//--%2F/g; s/\//--%2F/g' | sed 's/\//--%2F/g')
        SAFE_NAME2=$(echo "$REPO" | sed 's/\//--/g')

        found=0
        for pattern in "models--${SAFE_NAME2}" "models--${SAFE_NAME}"; do
            for dir in "$HF_CACHE"/$pattern "$HF_CACHE"/hub/$pattern 2>/dev/null; do
                if [ -d "$dir" ]; then
                    size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
                    echo "Found: $dir ($size)"
                    echo "Delete? [y/N]"
                    read -r confirm
                    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                        rm -rf "$dir"
                        echo "Deleted."
                    fi
                    found=1
                fi
            done
        done
        if [ $found -eq 0 ]; then
            echo "Model not found in cache: $REPO"
        fi
        ;;

    cleanup)
        echo "=== Cleanup Suggestions ==="
        echo ""
        echo "Largest cached models:"
        for dir in "$HF_CACHE"/models--* "$HF_CACHE"/hub/models--* 2>/dev/null; do
            [ -d "$dir" ] || continue
            model_name=$(basename "$dir" | sed 's/models--//' | sed 's/--/\//g')
            size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
            echo "  $size  $model_name"
        done | sort -rh | head -20
        echo ""
        echo "To delete: scripts/manage_cache.sh delete <org/model>"
        echo ""
        echo "vLLM compilation cache (safe to clear):"
        du -sh "$VLLM_CACHE" 2>/dev/null
        echo "  Clear with: rm -rf $VLLM_CACHE/*"
        ;;

    *)
        echo "Usage: scripts/manage_cache.sh {status|list|cleanup|delete <repo>}"
        exit 1
        ;;
esac
