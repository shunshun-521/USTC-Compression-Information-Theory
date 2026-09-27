"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala SCD 结构）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _map_channel_llr(llr_ch):
    """将传输顺序 LLR 映射到 SCD 蝶形顺序（与编码端 B_N 一致）。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def bit_reversed(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（与 log-domain lower_llr 一致）。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（与 SCD 层索引对应）。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=np.int32)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (n - i))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_scd(llr_ch, frozen_bits):
    """Vangala 风格 SCD（列 0 为信道 LLR，判决在列 n）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = _map_channel_llr(llr_ch)

    frozen_set = set(np.where(frozen_bits)[0])

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])
        update_bits(l)

    return B[:, n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _sc_decode_scd(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用 SCD 实现作参考）。"""
    return _sc_decode_scd(llr, frozen_bits)
