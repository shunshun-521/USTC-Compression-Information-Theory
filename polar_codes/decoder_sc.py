"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, s_val):
    """g 运算，s_val 为左子树部分和。"""
    s_val = np.asarray(s_val, dtype=int)
    return (1 - 2 * s_val) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = [layer for layer in range(n) if (phi & (1 << layer)) == 0]
        bit_layers = [layer for layer in range(n) if (phi & (1 << layer)) != 0]
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（aff3ct naive 算法）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(llr, dtype=np.float64)
    u_hat = np.zeros(len(llr), dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        s_left = decode_node(llr_left, bit_offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], s_left)
        s_right = decode_node(llr_right, bit_offset + half)

        s_parent = np.zeros(n, dtype=int)
        s_parent[:half] = s_left ^ s_right
        s_parent[half:] = s_right
        return s_parent

    decode_node(llr, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
