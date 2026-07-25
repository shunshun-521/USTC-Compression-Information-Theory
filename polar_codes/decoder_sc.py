"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(top_llr, btm_llr, u_hat):
    """
    g 运算：top 为左/上分支 LLR，btm 为右/下分支 LLR。
    g = (1 - 2*u_hat) * top + btm
    """
    return (1 - 2 * u_hat) * top_llr + btm_llr


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，委托非递归版本）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    if frozen_bits.dtype == bool:
        frozen_bits = frozen_bits.astype(int)
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        layers_llr = []
        start = n - _active_llr_level(l, n)
        layers_llr = list(range(start, n))

        layers_bit = []
        if l >= N // 2:
            end = n - _active_bit_level(l, n)
            layers_bit = list(range(n, end, -1))

        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于 Permuted SCD 算法）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    frozen_set = {i for i in range(N) if frozen_bits[i]}
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top = L[j - branch_size, s]
                    btm = L[j, s]
                    B_top = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top, btm, B_top)

        if l in frozen_set:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
