"""
极化码 SC（串行抵消）译码器
惰性 LLR 计算（与 in-place 蝶形编码器配套，自然序译码）
"""
import math
import numpy as np


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return La * (1.0 - 2.0 * u_hat) + Lb


def _B_check(ll, ii):
    return (ii // (1 << ll)) % 2


def _s_updater(ll, ii, s):
    if _B_check(ll - 1, ii):
        s[ll, ii] = s[ll - 1, ii]
    else:
        if s[ll - 1, ii] == -1:
            _s_updater(ll - 1, ii, s)
        j = ii + (1 << (ll - 1))
        if s[ll - 1, j] == -1:
            _s_updater(ll - 1, j, s)
        s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, j]


def _Li(ll, ii, llrs, s, n):
    if llrs[ll, ii] != -np.inf:
        return llrs[ll, ii]
    if _B_check(ll, ii) == 0:
        llrs[ll, ii] = f_operation(
            _Li(ll + 1, ii, llrs, s, n),
            _Li(ll + 1, ii + (1 << ll), llrs, s, n),
        )
    else:
        if ll > 0:
            _s_updater(ll, ii - (1 << ll), s)
        llrs[ll, ii] = g_operation(
            _Li(ll + 1, ii - (1 << ll), llrs, s, n),
            _Li(ll + 1, ii, llrs, s, n),
            s[ll, ii - (1 << ll)],
        )
    return llrs[ll, ii]


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = -np.ones((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for ii in range(N):
        if frozen_bits[ii]:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
            u_hat[ii] = 0
        else:
            llrs[0, ii] = _Li(0, ii, llrs, s, n)
            u_hat[ii] = 1 if llrs[0, ii] < 0 else 0
            s[0, ii] = u_hat[ii]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
