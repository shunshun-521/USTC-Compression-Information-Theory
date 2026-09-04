"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（upper branch / box-plus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（lower branch）：
    g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    等价于 u=0 时 La+Lb，u=1 时 -La+Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _frozen_mask(frozen_bits):
    return np.asarray(frozen_bits, dtype=bool)


def _prepare_channel_llr(llr_ch, N):
    """
    编码端含比特倒序置换时，将信道 LLR 映射到译码树自然顺序。
    L[i] = llr_ch[bit_reverse(i)]
    """
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


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


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = _prepare_channel_llr(llr, len(llr))
    frozen = _frozen_mask(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] ^ B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in decode_order:
        update_llrs(l)
        if frozen[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        update_bits(l)

    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与递归实现一致的层调度）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    llr_layer_vec = []
    bit_layer_vec = []

    for l in decode_order:
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l >= N // 2:
            bit_start = n - _active_bit_level(l, n) + 1
            bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
        else:
            bit_layer_vec.append([])

    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于分层 LLR/比特数组）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    lambda_offset, llr_layer_vec, bit_layer_vec, decode_order = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = _prepare_channel_llr(llr_ch, N)

    u_hat = np.zeros(N, dtype=int)

    for idx, l in enumerate(decode_order):
        for s in llr_layer_vec[idx]:
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if frozen[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        for s in bit_layer_vec[idx]:
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] ^ B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
