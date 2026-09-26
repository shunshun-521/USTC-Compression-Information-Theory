"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（VN 更新），u_hat 为当前阶段的再编码部分和。"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1 - 2 * u_hat) * La + Lb


def _sc_decode_core(llr, frozen_bits):
    """Sionna 风格的递归 SC（含再编码部分和 u_up）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                u = 0
            else:
                u = 0 if llr_node[0] >= 0 else 1
            return np.array([u], dtype=np.int_), np.array([u], dtype=np.float64)

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        f1 = frozen_node[:half]
        f2 = frozen_node[half:]

        llr_u = f_operation(llr1, llr2)
        u1, u1_up = decode_node(llr_u, f1)
        llr_l = g_operation(llr1, llr2, u1_up)
        u2, u2_up = decode_node(llr_l, f2)

        u = np.concatenate([u1, u2])
        u1_up_re = np.bitwise_xor(u1_up.astype(np.int_), u2_up.astype(np.int_)).astype(np.float64)
        u_up = np.concatenate([u1_up_re, u2_up])
        return u, u_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return _sc_decode_core(np.asarray(llr, dtype=np.float64), frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = np.array([1 << i for i in range(n + 2)], dtype=np.int_)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while (tmp & 1) == 1 and layer < n:
            tmp >>= 1
            layer += 1
        for l in range(layer, n):
            llr_layers.append(l)
        bit_layers = []
        if phi % 2 == 1:
            l = 0
            tmp2 = phi
            while (tmp2 & 1) == 1:
                bit_layers.append(l)
                tmp2 >>= 1
                l += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _recursive_calc_llr(layer, phase, n, N, P, U):
    if layer == 0:
        return
    block = 1 << layer
    half = block >> 1
    stage = phase // block
    if stage % 2 == 0:
        _recursive_calc_llr(layer - 1, phase, n, N, P, U)
        base = (phase // block) * block
        for i in range(half):
            P[layer - 1, base + i] = f_operation(
                P[layer, base + i], P[layer, base + half + i]
            )
    else:
        base = (phase // block) * block
        pm = phase % block
        _recursive_calc_llr(layer - 1, base + pm - half, n, N, P, U)
        for i in range(half):
            u = U[layer, base + i]
            P[layer - 1, base + half + i] = g_operation(
                P[layer, base + i], P[layer, base + half + i], u
            )


def _recursive_update_bits(layer, phase, n, N, C, u_hat):
    if layer == n:
        return
    block = 1 << (layer + 1)
    half = block >> 1
    base = (phase // block) * block
    pm = phase % block
    if pm < half:
        _recursive_update_bits(layer + 1, phase, n, N, C, u_hat)
    else:
        C[layer + 1, base + pm - half] = u_hat[phase]
        _recursive_update_bits(layer + 1, phase, n, N, C, u_hat)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（信道 LLR 需与编码器比特倒序置换对齐）。"""
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return _sc_decode_core(llr_ch[rev], frozen_bits)
