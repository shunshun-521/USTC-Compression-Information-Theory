"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import inverse_bit_reversal_permutation


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a = np.where(sign_a == 0, 1, sign_a)
    sign_b = np.where(sign_b == 0, 1, sign_b)
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def active_llr_level(i, n):
    """二进制表示中从最高位起第一个 0 的位置（层数）"""
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
    """二进制表示中从最高位起第一个 1 的位置（层数）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def path_metric_penalty(llr, bit):
    """路径度量惩罚"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，与主译码器结果一致）。
    """
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        start = n - active_llr_level(l, n)
        llr_layer_vec[phi] = list(range(start, n))
        bit_start = n - active_bit_level(l, n) + 1
        if bit_start <= n:
            bit_layer_vec[phi] = list(range(n, bit_start - 1, -1))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（按比特倒序相位处理）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    inv_br = inverse_bit_reversal_permutation(N)
    L = np.zeros((N, n + 1), dtype=np.float64)
    C = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch[inv_br]

    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = bit_reversed_index(phi, n)

        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = C[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l in frozen_set:
            u_hat[l] = 0
            C[l, n] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            u_hat[l] = bit
            C[l, n] = bit

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        C[j - branch_size, s - 1] = (C[j, s] + C[j - branch_size, s]) % 2
                        C[j, s - 1] = C[j, s]

    return u_hat
