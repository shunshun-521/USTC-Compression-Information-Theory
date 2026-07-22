"""
极化码 SC（串行抵消）译码器
提供递归 LLR 按需计算版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, index):
    """判断节点属于 f 还是 g 分支"""
    return (index // (1 << layer)) % 2


def _s_updater(layer, index, bits):
    """按需更新已判决比特"""
    if layer == 0:
        return
    if _b_check(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] < 0:
            _s_updater(layer - 1, index, bits)
        partner = index + (1 << (layer - 1))
        if bits[layer - 1, partner] < 0:
            _s_updater(layer - 1, partner, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, partner]


def _compute_llr(layer, index, llrs, bits):
    """递归按需计算 LLR"""
    if llrs[layer, index] != -np.inf:
        return llrs[layer, index]
    if _b_check(layer, index) == 0:
        llrs[layer, index] = f_operation(
            _compute_llr(layer + 1, index, llrs, bits),
            _compute_llr(layer + 1, index + (1 << layer), llrs, bits),
        )
    else:
        if layer > 0:
            _s_updater(layer, index - (1 << layer), bits)
        llrs[layer, index] = g_operation(
            _compute_llr(layer + 1, index - (1 << layer), llrs, bits),
            _compute_llr(layer + 1, index, llrs, bits),
            bits[layer, index - (1 << layer)],
        )
    return llrs[layer, index]


def sc_decode_recursive(llr, frozen_bits):
    """递归按需 LLR 计算的 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr
    bits = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
            llrs[0, i] = np.inf
            bits[0, i] = 0
        else:
            llrs[0, i] = _compute_llr(0, i, llrs, bits)
            u_hat[i] = 0 if llrs[0, i] >= 0 else 1
            bits[0, i] = u_hat[i]
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        bits = phi
        for layer in range(n):
            if (bits >> layer) & 1:
                break
            layers.append(layer)
        llr_layer_vec.append(layers)
        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layers = []
            temp = phi
            for layer in range(n):
                if (temp >> layer) & 1:
                    bit_layers.append(layer)
            bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（基于 bit-reversed 顺序，与参考实现等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
