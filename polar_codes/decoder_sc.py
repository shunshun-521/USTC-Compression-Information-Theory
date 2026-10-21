"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

from _scd_core import PermutedSCD, bit_reversed


def f_operation(La, Lb):
    sign = np.where(La >= 0, 1.0, -1.0) * np.where(Lb >= 0, 1.0, -1.0)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen = set(np.where(frozen_bits)[0].tolist())
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return PermutedSCD(len(llr_ch), llr_ch).decode(frozen)


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    import math

    n = int(math.log2(N))
    return None, [bit_reversed(i, n) for i in range(N)], None
