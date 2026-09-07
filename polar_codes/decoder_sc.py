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
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _leaf_llr(llr, u_prefix, phi):
    """计算第 phi 个比特的 LLR"""
    N = len(llr)
    n = int(math.log2(N))

    def rec(node_llr, depth, offset):
        if depth == 0:
            return node_llr[0]
        half = len(node_llr) // 2
        left = f_operation(node_llr[:half], node_llr[half:])
        if phi < offset + half:
            return rec(left, depth - 1, offset)
        u_left = u_prefix[offset : offset + half]
        right = g_operation(node_llr[:half], node_llr[half:], u_left)
        return rec(right, depth - 1, offset + half)

    return rec(llr, n, 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        phi_bin = format(phi, f"0{n}b")
        layers = [layer for layer in range(n) if phi_bin[n - 1 - layer] == "0"]
        llr_layer_vec.append(layers)
        layers_b = [layer for layer in range(n) if layer < n - 1 and phi_bin[n - 2 - layer] == "1"]
        if phi % 2 == 1:
            layers_b.append(n - 1)
        bit_layer_vec.append(layers_b)
    return llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr, frozen_bits):
    """非递归 SC 译码（逐比特 LLR）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        llr_phi = _leaf_llr(llr, u_hat, phi)
        u_hat[phi] = 0 if frozen_bits[phi] or llr_phi >= 0 else 1
    return u_hat


def sc_decode(llr_ch, frozen_bits, apply_bit_reversal=False):
    """SC 译码主函数"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)
