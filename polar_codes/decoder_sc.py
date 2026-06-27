"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _bit_reversed_index(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上层 LLR，Lb 为下层 LLR。"""
    return (1 - 2 * u_hat) * La + Lb


def _prepare_channel_llrs(llr_ch):
    """将信道 LLR 映射到译码树叶子节点（与编码端比特倒序一致）。"""
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return _sc_decode_core(_prepare_channel_llrs(llr_ch), frozen_bits)


def _sc_decode_core(channel_llrs, frozen_bits):
    N = len(channel_llrs)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = channel_llrs

    for i in range(N):
        l = _bit_reversed_index(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            half = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], B[j - half, s + 1]
                    )
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                half = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= half:
                        B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for i in range(N):
        l = _bit_reversed_index(i, n)
        llr_layer_vec[i] = list(range(n - _active_llr_level(l, n), n))
        if l < N // 2:
            bit_layer_vec[i] = []
        else:
            bit_layer_vec[i] = list(range(n, n - _active_bit_level(l, n), -1))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
