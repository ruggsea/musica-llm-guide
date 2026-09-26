#!/bin/bash
#SBATCH -N 1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH --exclusive
#SBATCH --time=00:40:00
set -eo pipefail
source /data/fs201045/rl41113/vllm-025-venv/bin/activate
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
rm -rf ~/.cache/flashinfer/0.6.13/90a/cached_ops/rope ~/.cache/flashinfer/0.6.13/90a/cached_ops/tmp
python - <<'PY'
import traceback
mods = []
try:
    from flashinfer.jit import gen_rope_module; mods.append(("rope", gen_rope_module))
except Exception: 
    try:
        from flashinfer.rope import gen_rope_module; mods.append(("rope", gen_rope_module))
    except Exception: traceback.print_exc()
for name in ["gen_norm_module","gen_activation_module","gen_quantization_module","gen_page_module","gen_sampling_module"]:
    try:
        import flashinfer.jit as J
        fn = getattr(J, name, None)
        if fn: mods.append((name, fn))
    except Exception: pass
for name, fn in mods:
    try:
        m = fn(); m.build_and_load(); print("BUILT", name)
    except Exception as e:
        print("SKIP", name, repr(e)[:120])
PY
ls ~/.cache/flashinfer/0.6.13/90a/cached_ops/rope/ 2>/dev/null | head -3
echo WARM-DONE
