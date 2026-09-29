"""
极化码 SC（串行抵消）译码器
Permuted SCD（Vangala 2014），log-domain f/g
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def boxplus_exact(la, lb):
    la = np.asarray(la, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    if la.size == 1 and lb.size == 1:
        return upper_llr(float(la), float(lb))
    out = np.empty_like(la)
    for i in range(la.size):
        out.flat[i] = upper_llr(la.flat[i], lb.flat[i])
    return out


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed_index(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return list(range(N)), llr_layer_vec, bit_layer_vec


def _scd_decode(llr_ch, frozen_set):
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(int(i) for i in frozen_set)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(top_bit)
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    frozen = np.where(np.asarray(frozen_bits).astype(bool))[0]
    return _scd_decode(np.asarray(llr, dtype=np.float64), frozen)


def sc_decode(llr_ch, frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype != bool:
        frozen_set = np.where(fb != 0)[0]
    else:
        frozen_set = np.where(fb)[0]
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _scd_decode(np.asarray(llr_ch, dtype=np.float64)[br], frozen_set)
