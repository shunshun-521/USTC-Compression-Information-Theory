"""加载 vendored SCD 实现（无 tkinter 依赖）"""
import importlib.util
import sys

_SCD = None


def _load_scd():
    global _SCD
    if _SCD is not None:
        return _SCD
    base = "/workspace/polar_codes/_ref"
    for name in ["utils", "decoder_utils", "SCD"]:
        path = f"{base}/{name}.py"
        if name == "SCD":
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            _SCD = mod.SCD
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
    return _SCD


def run_scd(llrs, frozen_indices, N):
    SCD = _load_scd()
    n = int(__import__("math").log2(N))

    class _PC:
        pass

    pc = _PC()
    pc.N = N
    pc.n = n
    pc.frozen = list(frozen_indices)
    pc.likelihoods = llrs
    return SCD(pc).decode()
