"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 PSCD 实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：下分支 LLR = btm + top (u=0) 或 btm - top (u=1)
    参数顺序：La=top, Lb=btm
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(i, n):
    rev = 0
    for b in range(n):
        if (i >> b) & 1:
            rev |= 1 << (n - 1 - b)
    return rev


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
    N = L.shape[0]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top, btm)
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top, btm, top_bit)


def _update_bits(B, l, n):
    if l < (1 << (n - 1)):
        return
    N = B.shape[0]
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（PSCD）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=np.int8)
    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        _update_bits(B, l, n)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托 PSCD 实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
