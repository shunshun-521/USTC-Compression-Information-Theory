"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

_LLR_MAX = 30.0


def f_operation(La, Lb):
    """Check-node boxplus（向量化，稳定 tanh 形式）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    a = np.clip(La, -_LLR_MAX, _LLR_MAX)
    b = np.clip(Lb, -_LLR_MAX, _LLR_MAX)
    t = np.tanh(a / 2.0) * np.tanh(b / 2.0)
    t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(t)


def g_operation(La, Lb, u_hat):
    """Variable-node (g) 运算"""
    return (1 - 2 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _llr_to_bit(llr, frozen):
    if frozen:
        return 0
    if llr == 0:
        return 1
    return int(llr < 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（Sionna 树结构 + 中间 re-encode）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(L, frozen_seg):
        n = len(L)
        if n == 1:
            u = _llr_to_bit(L[0], frozen_seg[0])
            return np.array([u], dtype=int), np.array([u], dtype=int)

        half = n // 2
        L_left = f_operation(L[:half], L[half:])
        u_left, u_left_up = decode_node(L_left, frozen_seg[:half])

        L_right = g_operation(L[:half], L[half:], u_left_up)
        u_right, u_right_up = decode_node(L_right, frozen_seg[half:])

        u_hat = np.concatenate([u_left, u_right])
        u_left_up = u_left_up.astype(np.int8)
        u_right_up = u_right_up.astype(np.int8)
        left_xor = np.bitwise_xor(u_left_up, u_right_up)
        u_up = np.concatenate([left_xor, u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 所需辅助向量（与经典相位更新一致）"""
    n = int(math.log2(N))
    lambda_offset = [1 << d for d in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        while p & 1:
            layers_llr.append((p.bit_length() - 1))
            p >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            layers_bit.append(0)
        pp = phi >> 1
        while pp & 1:
            layers_bit.append(pp.bit_length() - 1)
            pp >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：当前以递归实现为后端（正确性优先）"""
    return sc_decode_recursive(np.asarray(llr_ch, dtype=np.float64), frozen_bits)
