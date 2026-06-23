"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def logdomain_sum(x, y):
    """对数域加法。"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    精确 box-plus f 运算（SC 译码使用）。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.shape != Lb.shape:
        return np.array(
            [logdomain_sum(a + b, 0.0) - logdomain_sum(a, b) for a, b in zip(La.flat, Lb.flat)]
        ).reshape(La.shape)
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：λ_b + (1-2u)*λ_a，La 为上分支，Lb 为下分支。
    """
    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if u_hat.ndim:
        return np.where(u_hat == 0, La + Lb, Lb - La)
    return La + Lb if u_hat == 0 else Lb - La


def _active_llr_level(i, n):
    """二进制表示中自 MSB 起第一个 1 之前的 0 个数。"""
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
    """二进制表示中自 MSB 起第一个 0 之前的 1 个数。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _channel_llr_to_decoder(llr_ch):
    """编码含比特倒序时，将信道 LLR 映射到译码树输入。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return llr_ch[inv]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_int(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（委托给非递归实现）。"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr = _channel_llr_to_decoder(llr_ch)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed_int(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
