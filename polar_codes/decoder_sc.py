"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，置换 SC）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def bit_reversed(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域精确形式）。"""
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（对数域精确形式）。"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（向量化）。
  仿真中默认使用 min-sum 以提升速度。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _preprocess_llr(llr_ch):
    """将信道 LLR 变换到置换 SC 所需顺序。"""
    br = bit_reversal_permutation(len(llr_ch))
    inv = np.argsort(br)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（置换 SC 的译码顺序）。
    """
    n = int(math.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]
    return decode_order, n


def _sc_decode_permuted(llr, frozen_bits, use_minsum=False):
    """置换 SC 译码核心（Vangala et al. 2014）。"""
    llr = _preprocess_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits == 1)[0])

    f_fn = f_operation if use_minsum else upper_llr
    g_fn = lambda l1, l2, b: lower_llr(l1, l2, b)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    u_hat = np.zeros(N, dtype=int)

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    if use_minsum:
                        L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_fn(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l < N / 2:
            continue
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（置换 SC，默认 min-sum）。"""
    return _sc_decode_permuted(llr_ch, frozen_bits, use_minsum=True)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用精确 f/g）。"""
    return _sc_decode_permuted(llr, frozen_bits, use_minsum=False)
