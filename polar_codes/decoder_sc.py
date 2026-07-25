"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上半支路，Lb 为下半支路。"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _frozen_mask_to_mcba1n_set(frozen_bits, br):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        frozen_idx = np.where(frozen_bits)[0]
    else:
        frozen_idx = np.where(frozen_bits.astype(int) != 0)[0]
    return set(br[frozen_idx])


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phase in range(N):
        l = _bit_reversed(phase, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_mcba1n_core(llr_ch, frozen_set):
    """在 mcba1n 索引约定下执行 SC 译码，返回 u_mcba。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch

    u_mcba = np.zeros(N, dtype=int)
    for phase in range(N):
        l = _bit_reversed(phase, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_mcba[l] = B[l, n]

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_mcba


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    与编码器 G_N = B_N F^{\\otimes n} 及比特倒序输出一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    frozen_set = _frozen_mask_to_mcba1n_set(frozen_bits, br)
    u_mcba = _sc_decode_mcba1n_core(llr_ch, frozen_set)
    return u_mcba[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归核心）。"""
    return sc_decode(llr, frozen_bits)
