"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """log-domain f（min-sum 近似）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """log-domain g"""
    if np.isnan(u_hat):
        u_hat = 0
    u_hat = int(u_hat)
    if u_hat == 0:
        return La + Lb
    return La - Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n - 1, -1, -1)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llr(L, B, x, n):
    for j in range(n - 1, -1, -1):
        s = 2 ** (n - j)
        t = s // 2
        for i in range(x, len(L), s):
            if t > (i % s):
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, x, n):
    b = [x]
    for j in range(n):
        s = 2 ** (n - j)
        t = s // 2
        bnext = []
        for i in b:
            if t <= (i % s):
                b_left = 0 if np.isnan(B[i - t, j]) else int(B[i - t, j])
                b_i = 0 if np.isnan(B[i, j]) else int(B[i, j])
                B[i - t, j + 1] = b_i ^ b_left
                B[i, j + 1] = B[i, j]
                bnext.extend([i, i - t])
        b = bnext


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Algorithms 3–5）。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    br = bit_reversal_permutation(N)
    llr = llr_ch[br]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)

    L[:, n] = llr

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llr(L, B, l, n)

        if frozen_bits[l]:
            B[l, 0] = 0
        elif L[l, 0] >= 0:
            B[l, 0] = 0
        else:
            B[l, 0] = 1

        _update_bits(B, l, n)

    return np.nan_to_num(B[:, 0], nan=0.0).astype(int)
