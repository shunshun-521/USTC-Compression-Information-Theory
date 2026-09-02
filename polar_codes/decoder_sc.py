"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _sc_decode_core(llr, frozen_bits):
    """递归 SC 核心（与蝶形编码 + 比特倒序置换配对）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                u = 0
            else:
                u = 0 if llr_node[0] >= 0 else 1
            return np.array([u], dtype=int), np.array([u], dtype=int)

        half = n // 2
        llr_left = llr_node[:half]
        llr_right = llr_node[half:]
        frozen_left = frozen_node[:half]
        frozen_right = frozen_node[half:]

        llr_up = f_operation(llr_left, llr_right)
        u_left, u_left_up = decode_node(llr_up, frozen_left)

        llr_down = g_operation(llr_left, llr_right, u_left_up)
        u_right, u_right_up = decode_node(llr_down, frozen_right)

        u_hat = np.concatenate([u_left, u_right])
        u_up = np.concatenate([u_left_up ^ u_right_up, u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_node(np.asarray(llr, dtype=np.float64), frozen_bits)
    return u_hat


def _permute_llr(llr_ch):
    """将信道 LLR 置换为蝶形译码顺序。"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。frozen_bits: True 表示冻结位。"""
    return _sc_decode_core(_permute_llr(llr), frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供扩展使用）。"""
    n = int(np.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            p >>= 1
            llr_layers.append(len(llr_layers))
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        p = phi
        layer = 0
        while p & 1:
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（递归实现，O(N log N)）。
    frozen_bits: 1 或 True 表示冻结位。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
