"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """f 运算（log-domain box-plus）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _upper_llr(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or u_hat.shape == ():
        return _lower_llr(Lb, La, int(u_hat))
    out = np.empty_like(La, dtype=np.float64)
    for i in range(len(La)):
        out[i] = _lower_llr(Lb[i], La[i], int(u_hat[i]))
    return out


def _logdomain_sum(a, b):
    """log(exp(a) + exp(b))。"""
    if np.isscalar(a):
        if a == np.inf and b == np.inf:
            return np.inf
        if a == -np.inf:
            return b
        if b == -np.inf:
            return a
        m = max(a, b)
        return m + np.log1p(np.exp(-abs(a - b)))
    m = np.maximum(a, b)
    return m + np.log1p(np.exp(-np.abs(a - b)))


def _upper_llr(l1, l2):
    """f 分支 LLR 更新。"""
    if np.isscalar(l1):
        if l1 == np.inf and l2 != np.inf:
            return l2
        if l1 != np.inf and l2 == np.inf:
            return l1
        if l1 == np.inf and l2 == np.inf:
            return np.inf
        return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)
    out = np.empty_like(l1, dtype=np.float64)
    for i in range(len(l1)):
        out[i] = _upper_llr(l1[i], l2[i])
    return out


def _lower_llr(l1, l2, b):
    """g 分支 LLR 更新。"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _update_bits(B, l, n):
    N = B.shape[0]
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n)
        if frozen_bits[i]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = _bit_reversed(i, n)
        u_hat[i] = int(B[l, n])
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口，供 SCL 使用。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(_bit_reversed(phi, n), n), n)))
        l = _bit_reversed(phi, n)
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
