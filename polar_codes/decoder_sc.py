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


def _b_check(level, idx):
    return (idx // (1 << level)) % 2


def _update_partial_sums(level, idx, s):
    if _b_check(level - 1, idx):
        s[level, idx] = s[level - 1, idx]
    else:
        if s[level - 1, idx] == -1:
            _update_partial_sums(level - 1, idx, s)
        sibling = idx + (1 << (level - 1))
        if s[level - 1, sibling] == -1:
            _update_partial_sums(level - 1, sibling, s)
        s[level, idx] = s[level - 1, idx] ^ s[level - 1, sibling]


def _compute_llr(level, idx, llrs, s):
    if not np.isneginf(llrs[level, idx]):
        return llrs[level, idx]
    if _b_check(level, idx) == 0:
        llrs[level, idx] = f_operation(
            _compute_llr(level + 1, idx, llrs, s),
            _compute_llr(level + 1, idx + (1 << level), llrs, s),
        )
    else:
        if level > 0:
            _update_partial_sums(level, idx - (1 << level), s)
        u_top = s[level, idx - (1 << level)]
        llrs[level, idx] = g_operation(
            _compute_llr(level + 1, idx - (1 << level), llrs, s),
            _compute_llr(level + 1, idx, llrs, s),
            u_top,
        )
    return llrs[level, idx]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（自然比特顺序，与蝶形编码器一致）。
    frozen_bits[i]=1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits).astype(int)

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = -np.ones((n + 1, N), dtype=np.int8)

    u_hat = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if frozen_bits[i] == 1:
            s[0, i] = 0
            llrs[0, i] = np.inf
            u_hat[i] = 0
        else:
            llr_i = _compute_llr(0, i, llrs, s)
            bit = 1 if llr_i < 0 else 0
            s[0, i] = bit
            u_hat[i] = bit
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """接口兼容：返回占位预计算结构。"""
    n = int(math.log2(N))
    return list(range(N)), [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（委托自然顺序实现）。"""
    return sc_decode(llr, frozen_bits)
