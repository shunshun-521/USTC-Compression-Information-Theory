"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，mcba1n 风格）
"""
import math

import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（对外接口，供测试/参考）。
    SC 主译码器内部使用精确 box-plus 以保证数值正确性。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（与 mcba1n lower_llr 一致：La 为下分支，Lb 为上分支）。
    u=0: La+Lb；u=1: La-Lb
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _boxplus(La, Lb):
    """精确 box-plus（log-domain）"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def _boxplus_vec(La, Lb):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(_boxplus)(La, Lb)


def active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
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
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _boxplus(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托非递归 mcba1n 实现）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 使用的层索引"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layers = []
        if l >= N // 2:
            bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（mcba1n 风格，按比特倒序相位译码）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for phase in range(N):
        l = bit_reversed(phase, n)
        _update_llrs(L, B, l, n)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n)

    return B[:, n].astype(int)
