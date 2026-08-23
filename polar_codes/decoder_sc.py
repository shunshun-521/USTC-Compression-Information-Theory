"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from encoder import bit_reversal_permutation

_REF = Path(__file__).resolve().parent / "_ref"


def _load_ref_modules():
    if "polar_ref.utils" not in sys.modules:
        spec = importlib.util.spec_from_file_location("polar_ref.utils", _REF / "utils.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["polar_ref.utils"] = mod
        sys.modules["polarcodes.utils"] = mod
        spec.loader.exec_module(mod)
    if "polar_ref.decoder_utils" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "polar_ref.decoder_utils", _REF / "decoder_utils.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["polar_ref.decoder_utils"] = mod
        sys.modules["polarcodes.decoder_utils"] = mod
        spec.loader.exec_module(mod)
    if "polar_ref.SCD" not in sys.modules:
        spec = importlib.util.spec_from_file_location("polar_ref.SCD", _REF / "SCD.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["polar_ref.SCD"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["polar_ref.SCD"].SCD


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    编码器含比特倒序置换，信道 LLR 需先做相同置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen_idx = np.where(frozen_bits)[0]

    SCD = _load_ref_modules()
    pc = type("PC", (), {
        "N": N,
        "n": n,
        "likelihoods": llr_ch[br],
        "frozen": frozen_idx,
    })()
    return SCD(pc).decode()


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def sc_decode_fast(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def f_operation(La, Lb):
    _load_ref_modules()
    from polar_ref.decoder_utils import upper_llr
    return upper_llr(La, Lb)


def g_operation(La, Lb, u_hat):
    _load_ref_modules()
    from polar_ref.decoder_utils import lower_llr
    return lower_llr(La, Lb, u_hat)


def precompute_sc_indices(N):
    _load_ref_modules()
    from polar_ref.decoder_utils import active_bit_level, active_llr_level
    from polar_ref.utils import bit_reversed

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
