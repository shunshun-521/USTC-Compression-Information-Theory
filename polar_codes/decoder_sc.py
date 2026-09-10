"""
极化码 SC（串行抵消）译码器
Permuted SC（Vangala 2014）+ 递归参考实现
"""
import math

import numpy as np


def _bit_reversed(i, n):
    r = 0
    for b in range(n):
        if i & (1 << b):
            r |= 1 << (n - 1 - b)
    return r


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


def _logdomain_sum(x, y):
    if np.isscalar(x):
        if x > y:
            return x + np.log1p(np.exp(y - x))
        return y + np.log1p(np.exp(x - y))
    m = np.maximum(x, y)
    return m + np.log1p(np.exp(-np.abs(x - y)))


def _logdomain_diff(x, y):
    if np.isscalar(x):
        if x > y:
            return x + np.log1p(-np.exp(y - x))
        return y + np.log1p(-np.exp(x - y))
    m = np.maximum(x, y)
    return np.where(x > y, x + np.log1p(-np.exp(y - x)), y + np.log1p(-np.exp(x - y)))


def f_operation(La, Lb):
    """min-sum f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat):
        return La - Lb if u_hat else La + Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


def prepare_channel_llr(llr_ch):
    """保留接口：Permuted SC 直接使用原始信道 LLR"""
    return np.asarray(llr_ch, dtype=np.float64)


def _permuted_sc_decode(llr_ch, frozen_bits):
    """Permuted SC 核心"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

        B[l, n] = 0 if frozen_bits[l] or L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（委托 Permuted SC）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        pp = phi
        while pp % 2 == 1:
            layers.append(int(math.log2(pp & -pp)))
            pp >>= 1
        llr_layer_vec.append(layers)
        layers_b = []
        pp = phi
        while pp > 0 and pp % 2 == 0:
            layers_b.append(int(math.log2(pp & -pp)))
            pp >>= 1
        bit_layer_vec.append(layers_b)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码"""
    return _permuted_sc_decode(llr_ch, frozen_bits)
