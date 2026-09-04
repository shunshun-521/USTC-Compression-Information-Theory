"""
极化码 SC（串行抵消）译码器
Vangala 置换 SC 实现
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc_core import SCD, upper_llr, lower_llr, bit_reversed, active_llr_level, active_bit_level


def f_operation(La, Lb):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        return upper_llr(float(La), float(Lb))
    return np.array([upper_llr(a, b) for a, b in zip(La.flat, Lb.flat)]).reshape(La.shape)


def f_operation_minsum(La, Lb):
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def _map_channel_llr(llr_ch):
    N = len(llr_ch)
    perm = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[perm] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        br = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(br, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(br, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = np.where(np.asarray(frozen_bits, dtype=bool))[0]
    mapped = _map_channel_llr(llr_ch)
    return SCD(N, n, frozen_set, mapped).decode()


def sc_decode_incremental(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
