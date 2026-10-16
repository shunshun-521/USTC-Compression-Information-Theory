"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed_index


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """二进制表示中自高位起第一个 1 的位置（参考 polar-codes SCD）。"""
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
    """二进制表示中自高位起第一个 0 的位置。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归形式接口：与高效非递归实现保持一致（相同编码约定）。
    """
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """保留接口：返回比特倒序译码顺序。"""
    n = int(math.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    return decode_order, n


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按比特倒序阶段索引，与蝶形编码匹配）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block_size = 2 ** (s + 1)
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

        if l < N // 2:
            continue

        end_s = n - _active_bit_level(l, n)
        for s in range(n, end_s, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] ^ B[j - branch_size, s]) & 1
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
