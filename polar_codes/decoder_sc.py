"""
极化码 SC（串行抵消）译码器
因子图逐步 SC（主实现）与递归接口（参考）
"""
import math
import numpy as np
from sc_stepping import sc_decode_stepping


def f_operation(La, Lb):
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[]] * N, [[]] * N


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_set = set(np.where(frozen_bits == 1)[0])
    return sc_decode_stepping(np.asarray(llr_ch, dtype=np.float64), frozen_set)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
