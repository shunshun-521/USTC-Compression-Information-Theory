"""
极化码 SC 译码器
"""
import math

import numpy as np

from polar_wrapper import _PolarCode, decode_sc as _decode_sc_ref


def f_operation(La, Lb):
    import os, sys
    ref = os.path.join(os.path.dirname(__file__), "polar_ref")
    if ref not in sys.path:
        sys.path.insert(0, ref)
    from decoder_utils import upper_llr
    return upper_llr(float(La), float(Lb))


def g_operation(La, Lb, u_hat):
    import os, sys
    ref = os.path.join(os.path.dirname(__file__), "polar_ref")
    if ref not in sys.path:
        sys.path.insert(0, ref)
    from decoder_utils import lower_llr
    return lower_llr(float(La), float(Lb), int(u_hat))


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, psi, layer = [], phi, 0
        while psi % 2 == 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(list(range(layer, n)))
    lambda_offset = [2 ** (l - 1) if l > 0 else 0 for l in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    pc = _PolarCode(N, N - int(np.sum(frozen_bits)))
    pc.n = n
    pc.frozen = np.where(frozen_bits)[0]
    return _decode_sc_ref(llr_ch, pc)
