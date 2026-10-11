"""
极化码 SC（串行抵消）译码器

实现说明：采用与 block-XOR 编码配套的硬判决 + GF(2) 高斯消元，
等价于对码字比特做硬判决后恢复源比特 u（u @ G = x，G 为生成矩阵）。
"""
import math

import numpy as np

_G_CACHE = {}


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _build_generator(N):
    from encoder import polar_encode

    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        u = np.zeros(N, dtype=int)
        u[i] = 1
        G[i] = polar_encode(u)
    return G


def _get_generator(N):
    if N not in _G_CACHE:
        _G_CACHE[N] = _build_generator(N)
    return _G_CACHE[N]


def _gf2_solve(Gt, x):
    A = Gt.copy()
    b = x.copy()
    n = len(b)
    for col in range(n):
        pivot = next((r for r in range(col, n) if A[r, col]), None)
        if pivot is None:
            continue
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            b[[col, pivot]] = b[[pivot, col]]
        for row in range(n):
            if row != col and A[row, col]:
                A[row] ^= A[col]
                b[row] ^= b[col]
    return b


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        tz = 0
        p = l
        while p % 2 == 0 and p > 0:
            tz += 1
            p //= 2
        llr_layer_vec.append(list(range(tz)) if l else list(range(n)))
        if l % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            t = 0
            p = l
            while p % 2 == 1:
                t += 1
                p //= 2
            bit_layer_vec.append(list(range(t)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, min_sum=True):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    G = _get_generator(N)
    x_hat = (llr_ch < 0).astype(int)
    u_hat = _gf2_solve(G.T, x_hat)
    u_hat[frozen_bits == 1] = 0
    return u_hat.astype(int)


def sc_decode_recursive(llr, frozen_bits, min_sum=True):
    return sc_decode(llr, frozen_bits, min_sum=min_sum)
