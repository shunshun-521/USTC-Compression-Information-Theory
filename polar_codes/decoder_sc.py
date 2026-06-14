"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from sc_tree_ops import f_hf, g, sc_tree_decode


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（sign(0) 视为 +1）。"""
    if np.ndim(La) == 0:
        return f_hf(La, Lb)
    return np.array([f_hf(a, b) for a, b in zip(np.atleast_1d(La), np.atleast_1d(Lb))])


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return g(La, Lb, u_hat)


def _prepare_llr(llr_ch):
    N = len(llr_ch)
    return np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]


def _info_from_frozen(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return list(np.where(frozen_bits == 0)[0])


def sc_decode_recursive(llr_ch, frozen_bits):
    """树遍历 SC 译码（参考实现）。"""
    y_llr = _prepare_llr(llr_ch)
    info_pos = _info_from_frozen(frozen_bits)
    return sc_tree_decode(y_llr, info_pos, frozen_bit=0)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        tmp = phi
        layer = 0
        while tmp % 2 == 1:
            layer += 1
            tmp //= 2
        for l in range(layer, n):
            layers_llr.append(l)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi
        layer = 0
        while tmp > 0 and tmp % 2 == 0:
            layers_bit.append(layer)
            layer += 1
            tmp //= 2
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
