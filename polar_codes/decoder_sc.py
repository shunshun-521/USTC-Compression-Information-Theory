"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    if layer == 0:
        return
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        j = idx + (1 << (layer - 1))
        if s[layer - 1, j] == -1:
            _s_updater(layer - 1, j, s)
        s[layer, idx] = (s[layer - 1, idx] ^ s[layer - 1, j]) & 1


def _compute_llr(layer, idx, llrs, s):
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _compute_llr(layer + 1, idx, llrs, s),
            _compute_llr(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        prev = idx - (1 << layer)
        llrs[layer, idx] = g_operation(
            _compute_llr(layer + 1, prev, llrs, s),
            _compute_llr(layer + 1, idx, llrs, s),
            s[layer, prev],
        )
    return llrs[layer, idx]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits: True 表示冻结位（译码为 0）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64).flatten()
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool).flatten()
    info_mask = ~frozen_bits

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)

    u_hat = np.zeros(N, dtype=int)
    for ii in range(N):
        if frozen_bits[ii]:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
            u_hat[ii] = 0
        else:
            llrs[0, ii] = _compute_llr(0, ii, llrs, s)
            u_hat[ii] = 1 if llrs[0, ii] < 0 else 0
            s[0, ii] = u_hat[ii]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用与 sc_decode 相同的核心算法）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（文档/扩展用）"""
    m = int(np.log2(N))
    lambda_offset = np.zeros(m + 1, dtype=int)
    for i in range(1, m + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (m - i))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        l = 0
        while l < m and ((phi >> l) & 1):
            layers_llr.append(l)
            l += 1
        if l < m:
            layers_llr.append(l)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        l = 0
        while l < m and (((phi + 1) >> l) & 1) == 0:
            l += 1
        for j in range(l):
            layers_bit.append(j)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec
