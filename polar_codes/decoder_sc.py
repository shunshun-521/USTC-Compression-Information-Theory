"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed
from scd_ref import SCD


class _PC:
    __slots__ = ("N", "n", "frozen", "likelihoods")


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    llr_upper = f_operation(llr[0::2], llr[1::2])
    u_upper = sc_decode_recursive(llr_upper, frozen_bits[0::2])
    llr_lower = g_operation(llr[0::2], llr[1::2], u_upper)
    u_lower = sc_decode_recursive(llr_lower, frozen_bits[1::2])

    u_hat = np.empty(N, dtype=int)
    u_hat[0::2] = u_upper
    u_hat[1::2] = u_lower
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    return [bit_reversed(i, n) for i in range(N)], None, None


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    pc = _PC()
    pc.N = len(llr_ch)
    pc.n = int(math.log2(pc.N))
    pc.frozen = set(np.where(frozen_bits)[0])
    pc.likelihoods = llr_ch
    return SCD(pc).decode()
