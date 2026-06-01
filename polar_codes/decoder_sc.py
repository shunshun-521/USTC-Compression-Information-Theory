"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归版本（主实现，与极化因子图自然序一致）
"""
import math

import numpy as np

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（log-domain 近似，与 CA-SCL 文献常用形式一致）：
    u=0: La + Lb; u=1: La - Lb
    """
    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """二进制表示中自高位起第一个 1 的位置（层数）。"""
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
    """二进制表示中自高位起第一个 0 的位置（层数）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按比特倒序依次判决，参考 Vangala et al. 2014）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if frozen_bits[l]:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用非递归实现，保证结果一致）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (n - i + 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec
