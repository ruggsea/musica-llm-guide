# Job-local fix for vllm#53242 (MiMo-V2 FP8 fused-QKV block-scale sharding), applied at import time.
# Put this dir on PYTHONPATH for a MiMo-V2.5 (non-Pro) job only. The shared venv is NOT modified,
# because the same fix silently mis-loads MiMo-V2.5-Pro. Fails loudly if the source does not match.
import importlib.abc, importlib.util, sys

_TARGET = "vllm.model_executor.models.mimo_v2"
_OLD = "    scale_rows_per_group = s_full.shape[0] // num_kv_heads\n    qs, ks, vs = [], [], []\n    for g_idx in range(tp_rank * kv_heads_per_rank, (tp_rank + 1) * kv_heads_per_rank):\n        row_start = g_idx * rows_per_group\n        scale_row_start = g_idx * scale_rows_per_group\n        # Dequantize this group's weights.\n        w_g = w_full[row_start : row_start + rows_per_group].to(torch.float32)\n        s_g = s_full[scale_row_start : scale_row_start + scale_rows_per_group].to(\n            torch.float32\n        )\n        s_g_expanded = s_g.repeat_interleave(block, dim=0).repeat_interleave(\n            block, dim=1\n        )[:rows_per_group]\n        w_g_dequant = w_g * s_g_expanded\n"
_NEW = "    s_full_expanded = s_full.repeat_interleave(block, dim=0).repeat_interleave(\n        block, dim=1\n    )[: w_full.shape[0], : w_full.shape[1]]\n    qs, ks, vs = [], [], []\n    for g_idx in range(tp_rank * kv_heads_per_rank, (tp_rank + 1) * kv_heads_per_rank):\n        row_start = g_idx * rows_per_group\n        # Dequantize this group's weights.\n        w_g = w_full[row_start : row_start + rows_per_group].to(torch.float32)\n        w_g_dequant = w_g * s_full_expanded[\n            row_start : row_start + rows_per_group\n        ]\n"


class _Loader(importlib.abc.SourceLoader):
    def __init__(self, path):
        self.path = path

    def get_filename(self, name):
        return self.path

    def get_data(self, path):
        src = open(path).read()
        if _NEW not in src:
            n = src.count(_OLD)
            if n != 1:
                raise RuntimeError(f"mimo53242 overlay: expected 1 match in {path}, found {n}; vLLM changed")
            src = src.replace(_OLD, _NEW)
        print("mimo53242 overlay: patched " + path, file=sys.stderr, flush=True)
        return src.encode()


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        return importlib.util.spec_from_file_location(name, spec.origin, loader=_Loader(spec.origin))


sys.meta_path.insert(0, _Finder())
