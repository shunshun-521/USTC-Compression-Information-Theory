"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        ll = []
        for layer in range(n, 0, -1):
            if ((phi >> (layer - 1)) & 1) == 0:
                ll.append(layer)
        llr_layer_vec.append(ll)

        bb = []
        for layer in range(n):
            if (phi >> layer) & 1:
                bb.append(layer)
        bit_layer_vec.append(bb)

    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（树结构参考实现）。"""
    from polar_tree import tree_sc_decode

    return tree_sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（树结构实现）。"""
    from polar_tree import tree_sc_decode

    return tree_sc_decode(llr_ch, frozen_bits)
