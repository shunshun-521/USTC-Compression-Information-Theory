"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（供 SCL / 递归参考实现使用）。
    """
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    def logdomain_sum(x, y):
        if x > y:
            return x + np.log1p(np.exp(y - x))
        return y + np.log1p(np.exp(x - y))

    def upper_llr(l1, l2):
        return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)

    def lower_llr(l1, l2, b):
        b = int(b) if not np.isnan(b) else 0
        return l1 + l2 if b == 0 else l1 - l2

    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _sc_recursive_core(llr, frozen_bits):
    """递归 SC 核心，返回 (u_hat, u_hat_up)。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)

    if n == 1:
        if frozen_bits[0]:
            bit = 0
        else:
            bit = 0 if llr[0] >= 0 else 1
        return np.array([bit], dtype=int), np.array([float(bit)])

    half = n // 2
    l1 = llr[:half]
    l2 = llr[half:]

    u_left, u_left_up = _sc_recursive_core(
        f_operation(l1, l2), frozen_bits[:half]
    )
    u_right, u_right_up = _sc_recursive_core(
        g_operation(l1, l2, u_left_up), frozen_bits[half:]
    )

    u_hat = np.concatenate([u_left, u_right])
    left_up = np.bitwise_xor(u_left_up.astype(int), u_right_up.astype(int)).astype(np.float64)
    u_hat_up = np.concatenate([left_up, u_right_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    信道 LLR 为传输顺序，译码前先做比特倒序重排。
    """
    br = bit_reversal_permutation(len(llr))
    llr_br = np.asarray(llr, dtype=np.float64)[br]
    u_hat, _ = _sc_recursive_core(llr_br, frozen_bits)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 先经比特倒序重排，再按 py-polar 顺序逐位判决。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_ch = llr_ch[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return u_hat


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    return (
        [1 << layer for layer in range(n + 1)],
        [[s for s in range(n - _active_llr_level(phi, n), n)] for phi in range(N)],
        [[s for s in range(n, n - _active_bit_level(phi, n), -1)] for phi in range(N)],
    )
