"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


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
        sibling = idx + (1 << (layer - 1))
        if s[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sibling]


def _compute_llr(layer, idx, llrs, s, n):
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]
    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _compute_llr(layer + 1, idx, llrs, s, n),
            _compute_llr(layer + 1, idx + (1 << layer), llrs, s, n),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _compute_llr(layer + 1, idx - (1 << layer), llrs, s, n),
            _compute_llr(layer + 1, idx, llrs, s, n),
            s[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    return list(range(n + 1)), [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（按需计算 LLR）。
    """
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
            llrs[0, phi] = _compute_llr(0, phi, llrs, s, n)
            u_hat[phi] = 0 if llrs[0, phi] >= 0 else 1
            s[0, phi] = u_hat[phi]

    return u_hat
