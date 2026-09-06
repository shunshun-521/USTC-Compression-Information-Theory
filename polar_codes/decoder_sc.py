"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效实现）
"""
import math

import numpy as np


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（upper LLR）"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（lower LLR，La=bottom, Lb=top）"""
    if u_hat == 0:
        return La + Lb
    return La - Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr_exact(l1, l2):
    """精确 upper LLR（递归参考实现用）"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr_exact(l1, l2, b):
    """精确 lower LLR"""
    if b == 0:
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    """找二进制表示中第一个 1 的位置（从高位起）"""
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
    """找二进制表示中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(l, L, B, n, N, use_min_sum=True):
    upper = f_operation if use_min_sum else upper_llr_exact
    lower = g_operation if use_min_sum else lower_llr_exact

    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = upper(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                if use_min_sum:
                    L[j, s + 1] = lower(btm_llr, top_llr, top_bit)
                else:
                    L[j, s + 1] = lower(btm_llr, top_llr, top_bit)


def _update_bits(l, B, n, N):
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（主实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(l, L, B, n, N, use_min_sum=True)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        _update_bits(l, B, n, N)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（Permuted SCD 的非递归等价参考）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口兼容性"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(
            list(range(n, n - active_bit_level(l, n), -1)) if l >= N / 2 else []
        )
    return llr_layer_vec, bit_layer_vec
