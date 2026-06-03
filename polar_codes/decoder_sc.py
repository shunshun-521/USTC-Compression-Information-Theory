"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(llr, dtype=np.float64)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                u = 0
            else:
                u = 0 if llr_node[0] >= 0 else 1
            return np.array([u], dtype=int), np.array([u], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left, u_left_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, frozen_node[half:])

        u_hat = np.concatenate([u_left, u_right])
        u_up_left = u_left_up ^ u_right_up
        u_up = np.concatenate([u_up_left, u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            bit_layers = list(range(n))
        else:
            tmp = phi
            for layer in range(1, n):
                if tmp & 1:
                    bit_layers.append(layer - 1)
                tmp >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前委托递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
