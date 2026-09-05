"""Adapter around verbatim mcba1n SCD reference."""
import math
import sys
import types
import importlib.util
import numpy as np

# Load reference modules with fake polarcodes package
if "polarcodes" not in sys.modules or not hasattr(sys.modules["polarcodes"], "_polar_loaded"):
    pkg = types.ModuleType("polarcodes")
    sys.modules["polarcodes"] = pkg

    def _load_ref(name, filename):
        path = f"/workspace/polar_codes/mcba1n/{filename}"
        spec = importlib.util.spec_from_file_location(f"polarcodes.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"polarcodes.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _utils = _load_ref("utils", "utils_ref.py")
    _load_ref("decoder_utils", "decoder_utils_ref.py")
    _scd = _load_ref("SCD", "scd_ref.py")
    pkg._polar_loaded = True
    pkg.SCD = sys.modules["polarcodes.SCD"].SCD
else:
    _scd = sys.modules["polarcodes.SCD"]


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        frozen_idx = np.where(fb)[0]
    else:
        frozen_idx = np.where(fb != 0)[0]

    class _PC:
        pass

    pc = _PC()
    pc.N = N
    pc.n = n
    pc.frozen = frozen_idx
    pc.likelihoods = llr_ch
    return sys.modules["polarcodes.SCD"].SCD(pc).decode()


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def sc_decode_layered(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    from mcba1n.decoder_utils import active_bit_level, active_llr_level
    from mcba1n.utils import bit_reversed

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n - active_llr_level(bit_reversed(i, n), n), n)) for i in range(N)]
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)) if l >= N / 2 else [])
    return lambda_offset, llr_layer_vec, bit_layer_vec
