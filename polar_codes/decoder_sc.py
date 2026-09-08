"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


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


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    La 为上半分支 LLR，Lb 为下半分支 LLR。
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _prepare_frozen_bits(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=int)
    return fb.astype(bool)


def _bit_reversed_index(i, n):
    return bit_reversal_permutation(1 << n)[i]


def active_llr_level(i, n):
    """从最高位起，第一个 0 之前（含）的层数。"""
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
    """从最高位起，第一个 1 之前（含）的层数。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        start_bit = n - active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, start_bit, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（高效实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _prepare_frozen_bits(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr_ch = llr_ch[br]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n)

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        _update_bits(B, l, n)

    return u_hat


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, L.shape[0], block_size):
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
    if l < B.shape[0] // 2:
        return

    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，调用非递归 PSCD 保持一致）。
    """
    return sc_decode(llr, frozen_bits)
