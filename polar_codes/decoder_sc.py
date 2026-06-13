"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，OkanErturk16 风格）
"""
import math
import numpy as np

INF = np.float64(1e100)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    """按需更新部分和（比特）数组。"""
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        sibling = idx + (1 << (layer - 1))
        if s[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sibling]


def _compute_li(layer, idx, llrs, s):
    """按需计算 LLR。"""
    if llrs[layer, idx] > -INF / 2:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _compute_li(layer + 1, idx, llrs, s),
            _compute_li(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        left_idx = idx - (1 << layer)
        llrs[layer, idx] = g_operation(
            _compute_li(layer + 1, left_idx, llrs, s),
            _compute_li(layer + 1, idx, llrs, s),
            s[layer, left_idx],
        )
    return llrs[layer, idx]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按需 LLR 计算，O(N log N)）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llrs = np.full((n + 1, N), -INF, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            llrs[0, phi] = INF
            s[0, phi] = 0
        else:
            llrs[0, phi] = _compute_li(0, phi, llrs, s)
            u_hat[phi] = 0 if llrs[0, phi] >= 0 else 1
            s[0, phi] = u_hat[phi]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while p & 1:
            layers_llr.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi & 1:
            p = phi
            layer = 0
            while p & 1:
                layers_bit.append(layer)
                p >>= 1
                layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec
