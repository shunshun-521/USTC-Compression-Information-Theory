"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from channel import reorder_llr_for_decode


def bit_reversed(x, n):
    """比特倒序索引"""
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
    """f 运算（log-domain boxplus）"""
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """向量化的 f 运算（log-domain）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(upper_llr)(La, Lb)


def g_operation(La, Lb, u_hat):
    """向量化的 g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    return np.vectorize(lower_llr)(La, Lb, u_hat)


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


def _frozen_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int) == 1)[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（log-domain f/g）"""
    llr = reorder_llr_for_decode(llr)
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen = _frozen_set(frozen_bits)
    u_hat = np.zeros(N, dtype=int)

    def decode_rec(node, offset):
        size = len(node)
        if size == 1:
            idx = offset
            if idx in frozen:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if node[0] >= 0 else 1
            return
        half = size // 2
        llr_u = f_operation(node[:half], node[half:])
        decode_rec(llr_u, offset)
        u_left = u_hat[offset : offset + half]
        llr_v = g_operation(node[:half], node[half:], u_left)
        decode_rec(llr_v, offset + half)

    decode_rec(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SC / mcba1n 风格，适配含比特倒序的编码）。
    """
    llr_ch = reorder_llr_for_decode(llr_ch)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = _frozen_set(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
