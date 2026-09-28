# Job-local fix for MiMo-V2.5 FP8 fused-QKV sharding: upstream vLLM PR #57508 (merged 2026-09-19, merge commit
# 211e252d), which replaced PR #53242. The 53242 overlay made MiMo-V2.5 load but generate gibberish (2026-09-28).
# Put this dir on PYTHONPATH for a MiMo-V2.5 job. The shared venv is NOT modified.
# Pinned: only replaces vllm/model_executor/models/mimo_v2.py when the installed file is exactly the one from
# vllm 0.26.1rc1.dev1212+gd125b540b (md5 below); anything else raises, so a newer vLLM cannot be silently overridden.
# mimo_v2_pr57508.py = that file + pr57508_mimo_v2.diff (the PR's mimo_v2.py hunks; the MTP file is not patched).
import hashlib, importlib.abc, importlib.util, os, sys

_TARGET = "vllm.model_executor.models.mimo_v2"
_PINNED_MD5 = "133a8b747e92284cc55dee427d6fed38"
_PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mimo_v2_pr57508.py")


class _Loader(importlib.abc.SourceLoader):
    def __init__(self, path):
        self.path = path

    def get_filename(self, name):
        return self.path

    def get_data(self, path):
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        if md5 != _PINNED_MD5:
            raise RuntimeError(f"mimo57508 overlay: {path} md5 {md5} != pinned {_PINNED_MD5}; vLLM changed, "
                               "check whether it already contains PR 57508 and drop this overlay")
        print("mimo57508 overlay: replaced " + path, file=sys.stderr, flush=True)
        return open(_PATCHED, "rb").read()


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
