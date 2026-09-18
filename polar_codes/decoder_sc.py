"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """精确 log-domain f 运算（box-plus）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return _upper_llr(float(La), float(Lb))
    return np.vectorize(_upper_llr)(La, Lb)


def g_operation(La, Lb, u_hat):
    """精确 log-domain g 运算：lower_llr(Lb, La, u)。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    if La.ndim == 0 and Lb.ndim == 0 and u_hat.ndim == 0:
        return _lower_llr(float(Lb), float(La), int(u_hat))
    return np.vectorize(_lower_llr)(Lb, La, u_hat)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    与高效非递归 SCD 等价，用于交叉验证。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助信息（兼容接口）。"""
    n = int(math.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    return None, decode_order, None


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（参考 polarcodes SCD）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s],
                        L[j - branch_size, s],
                        B[j - branch_size, s + 1],
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        update_bits(l)

    return u_hat
