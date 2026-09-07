"""
极化码 SC（串行抵消）译码器
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from scd_core import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    scd_decode_natural,
    upper_llr,
)


def f_operation(La, Lb):
    if np.isscalar(La) and np.isscalar(Lb):
        return upper_llr(La, Lb)
    return np.vectorize(upper_llr)(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：lower_llr(bottom, top, bit)"""
    u_hat = np.asarray(u_hat)
    if np.isscalar(La) and np.isscalar(Lb) and np.isscalar(u_hat):
        return lower_llr(Lb, La, int(u_hat))
    return np.vectorize(lambda top, bottom, u: lower_llr(bottom, top, int(u)))(La, Lb, u_hat)


def _permute_channel_llr(llr_ch):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = np.zeros(N, dtype=np.float64)
    llr[br] = llr_ch
    return llr


def sc_decode_recursive(llr_ch, frozen_bits):
    llr = _permute_channel_llr(llr_ch)
    return scd_decode_natural(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(
            list(range(n, n - active_bit_level(l, n), -1)) if l >= N // 2 else []
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    llr = _permute_channel_llr(llr_ch)
    return scd_decode_natural(llr, frozen_bits)
