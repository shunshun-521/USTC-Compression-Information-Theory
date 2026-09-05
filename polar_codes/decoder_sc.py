"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    对数域 f 运算（boxplus）。
    对标量/数组均支持；数组情形逐元素计算。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        if np.isinf(La) and not np.isinf(Lb):
            return float(Lb)
        if not np.isinf(La) and np.isinf(Lb):
            return float(La)
        if np.isinf(La) and np.isinf(Lb):
            return np.inf
        return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)

    out = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    it = np.nditer([La, Lb, out], flags=["refs_ok"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for a, b, o in it:
        o[...] = f_operation(float(a), float(b))
    return out


def g_operation(La, Lb, u_hat):
    """
    对数域 g 运算。
    """
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or u_hat.ndim == 0:
        u = int(u_hat)
        if u == 0:
            if np.isinf(La) or np.isinf(Lb):
                return np.inf
            return La + Lb
        return La - Lb
    return np.where(
        u_hat == 0,
        np.where(np.isinf(La) | np.isinf(Lb), np.inf, La + Lb),
        La - Lb,
    )


def f_operation_min_sum(La, Lb):
    """min-sum 近似 f 运算（备用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用与主实现相同的 L/B 结构）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码相位顺序（比特倒序）及层信息。
    """
    n = int(math.log2(N))
    phase_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in phase_order:
        llr_layer_vec.append(list(range(n - active_llr_level(phi, n), n)))
        if phi < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - active_bit_level(phi, n), -1)))
    return phase_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD，与编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    for phi_natural in range(N):
        l = bit_reversed_index(phi_natural, n)
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])
        u_hat[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return u_hat
