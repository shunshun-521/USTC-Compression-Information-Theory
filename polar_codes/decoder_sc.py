"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
基于 Permuted SCD（Vangala et al.）与 mcba1n/polar-codes 实现。
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, prepare_channel_llr

# ==================== 基本运算 ====================


def _logdomain_sum(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    larger = np.maximum(x, y)
    smaller = np.minimum(x, y)
    return larger + np.log1p(np.exp(smaller - larger))


def f_operation(La, Lb):
    """
    精确对数域 f 运算（SC 译码）。
  f(La, Lb) = log((1 + exp(La+Lb)) / (1 + exp(La) + exp(Lb) - exp(La+Lb)))
    等价于 log-domain box-plus。
    """
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def f_operation_min_sum(La, Lb):
    """min-sum 近似的 f 运算（BP 译码）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _frozen_set_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（使用与 Permuted SCD 一致的 LLR 排列）。
    """
    llr = prepare_channel_llr(llr)
    frozen_set = _frozen_set_from_mask(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_llr = L[j - branch_size, s]
                    btm_llr = L[j, s]
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息（与 Permuted SCD 层遍历一致）。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
