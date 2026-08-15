"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，惰性 LLR 计算）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


_INF = np.inf


def _b_check(layer, idx):
    """判断节点 (layer, idx) 是否为 g 节点（下分支）"""
    return (idx // (1 << layer)) % 2 == 1


def _s_updater(layer, idx, s):
    """惰性计算并回传硬判决比特"""
    if layer <= 0:
        return
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        sibling = idx + (1 << (layer - 1))
        if s[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sibling]


def _compute_llr(layer, idx, llrs, s):
    """递归惰性计算 LLR"""
    if llrs[layer, idx] != -_INF:
        return llrs[layer, idx]

    if not _b_check(layer, idx):
        llrs[layer, idx] = f_operation(
            _compute_llr(layer + 1, idx, llrs, s),
            _compute_llr(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        top_idx = idx - (1 << layer)
        if layer > 0:
            _s_updater(layer, top_idx, s)
        llrs[layer, idx] = g_operation(
            _compute_llr(layer + 1, top_idx, llrs, s),
            _compute_llr(layer + 1, idx, llrs, s),
            s[layer, top_idx],
        )
    return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_block, layer, bit_start):
        if layer == 0:
            idx = bit_start
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_block[0] >= 0 else 1
            return np.array([u_hat[idx]])

        half = 1 << (layer - 1)
        L1 = llr_block[:half]
        L2 = llr_block[half:]
        left_bits = decode_block(f_operation(L1, L2), layer - 1, bit_start)
        right_bits = decode_block(
            g_operation(L1, L2, left_bits), layer - 1, bit_start + half
        )
        return np.concatenate([(left_bits + right_bits) % 2, right_bits])

    decode_block(llr, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(np.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)
        if phi % 2 == 1:
            layers_bit = []
            psi = phi
            while psi % 2 == 1:
                layers_bit.append(int(np.log2(psi & -psi)))
                psi >>= 1
            bit_layer_vec.append(layers_bit)
        else:
            bit_layer_vec.append(list(range(n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（惰性 LLR 计算，O(N log N)）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -_INF, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)

    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        llr_val = _compute_llr(0, phi, llrs, s)
        if frozen_bits[phi]:
            u_hat[phi] = 0
            s[0, phi] = 0
        else:
            u_hat[phi] = 0 if llr_val >= 0 else 1
            s[0, phi] = u_hat[phi]

    return u_hat
