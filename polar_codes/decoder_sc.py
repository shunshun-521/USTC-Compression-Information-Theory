"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def _bit_reversed_index(i, n):
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算: La=top, Lb=bottom"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, Lb - La)


def _logdomain_sum(a, b):
    if a == -np.inf and b == -np.inf:
        return -np.inf
    if a == np.inf:
        return np.inf
    if b == np.inf:
        return np.inf
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def _upper_llr_exact(l1, l2):
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                return np.array([0])
            return np.array([0 if llr_node[0] >= 0 else 1])
        half = n // 2
        left_llr = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_node(left_llr, frozen_node[:half])
        right_llr = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_node(right_llr, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = (1 << layer) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        br_phi = _bit_reversed_index(phi, n)
        for layer in range(n):
            if (br_phi >> layer) & 1 == 0:
                layers_llr.append(layer)
            else:
                break
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if br_phi % 2 == 1:
            for layer in range(n):
                if (br_phi >> layer) & 1:
                    layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（对数域 f 函数，按比特倒序译码）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    bottom_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    if top_bit == 0:
                        L[j, s + 1] = bottom_llr + top_llr
                    else:
                        L[j, s + 1] = bottom_llr - top_llr

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        end_s = n - _active_bit_level(l, n)
        for s in range(n, end_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
