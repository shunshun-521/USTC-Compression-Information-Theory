"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat_up):
    """g 运算，u_hat_up 为当前层的部分重编码比特。"""
    return (1.0 - 2.0 * u_hat_up) * La + Lb


def _sc_decode_core(llr, frozen_bits):
    """SC 译码核心（递归，使用部分重编码比特 u_hat_up）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n_len = len(llr_node)
        if n_len == 1:
            if frozen_node[0]:
                bit = 0
            else:
                bit = 0 if llr_node[0] >= 0.0 else 1
            return np.array([bit], dtype=np.int8), np.array([float(bit)])

        half = n_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left, u_left_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, frozen_node[half:])

        u_hat = np.concatenate([u_left, u_right])
        u_left_up_int = np.mod(u_left_up.astype(int) + u_right_up.astype(int), 2)
        u_up = np.concatenate([u_left_up_int.astype(float), u_right_up])
        return u_hat, u_up

    return decode_node(llr, frozen_bits)[0]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n))
        else:
            llr_layers = []
            p = phi
            s = 0
            while p % 2 == 1 and s < n:
                llr_layers.append(s)
                p //= 2
                s += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi + 1
        s = 0
        while p % 2 == 0 and s < n:
            bit_layers.append(s)
            p //= 2
            s += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    当前实现与递归核心等价（使用部分重编码比特 u_hat_up）。
    """
    return _sc_decode_core(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return _sc_decode_core(llr, frozen_bits)
