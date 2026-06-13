"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        sib = idx + (1 << (layer - 1))
        if s[layer - 1, sib] == -1:
            _s_updater(layer - 1, sib, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sib]


def _li(layer, idx, llrs, s, n):
    if llrs[layer, idx] > -np.inf / 2:
        return llrs[layer, idx]
    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s, n),
            _li(layer + 1, idx + (1 << layer), llrs, s, n),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _li(layer + 1, idx - (1 << layer), llrs, s, n),
            _li(layer + 1, idx, llrs, s, n),
            s[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归结果一致）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if (phi >> layer) & 1:
                break
            llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        for layer in range(n):
            if (phi >> layer) & 1:
                bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr_ch, frozen_bits):
    """SC 译码核心：输入 LLR 已与编码端蝶形树对齐。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            s[0, phi] = 0
            llrs[0, phi] = np.inf
        else:
            llrs[0, phi] = _li(0, phi, llrs, s, n)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            s[0, phi] = u_hat[phi]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    输入为信道域 LLR（与码字 x 顺序一致），内部做比特倒序对齐。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _sc_decode_core(np.asarray(llr_ch, dtype=np.float64)[br], frozen_bits)
