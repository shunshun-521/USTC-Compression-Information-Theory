"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np

from scd_vendor.SCD import SCD
from scd_vendor.decoder_utils import upper_llr, lower_llr, active_llr_level, active_bit_level


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


class _PC:
    def __init__(self, N, n, llr_ch, frozen_set):
        self.N = N
        self.n = n
        self.likelihoods = llr_ch
        self.frozen = set(int(x) for x in frozen_set)


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _trailing_ones(phi):
    c = 0
    while phi & 1:
        c += 1
        phi >>= 1
    return c


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = np.array([1 << i for i in range(n)], dtype=np.int64)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        u = _trailing_ones(phi)
        llr_layer_vec.append(list(range(n - u, n)))
        bit_layer_vec.append(list(range(u)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = np.where(frozen_bits)[0]
    pc = _PC(N, n, llr_ch, frozen_set)
    return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
