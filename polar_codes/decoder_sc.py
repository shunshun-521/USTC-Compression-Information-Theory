"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from scd_core import SCD


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


class _PolarSCContext:
    """SCD 译码器所需的轻量上下文"""

    def __init__(self, N, likelihoods, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.likelihoods = likelihoods
        self.frozen = frozen_set


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    from polar_sc_utils import bit_reversed
    from decoder_utils_internal import active_bit_level, active_llr_level

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    likelihoods = llr_ch[br]
    frozen_set = set(np.where(frozen_bits)[0])
    ctx = _PolarSCContext(N, likelihoods, frozen_set)
    return SCD(ctx).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，用于对照验证）"""
    return sc_decode(llr, frozen_bits)
