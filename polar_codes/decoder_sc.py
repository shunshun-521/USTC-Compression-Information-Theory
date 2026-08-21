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
    """g 运算。"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
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
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def align_llr_for_decoder(llr_ch, N):
    """编码端比特倒序后，译码前对信道 LLR 做相同置换。"""
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    half = N // 2
    lr_left = f_operation(llr[:half], llr[half:])
    u_left = sc_decode_recursive(lr_left, frozen_bits[:half])
    lr_right = g_operation(llr[:half], llr[half:], u_left)
    u_right = sc_decode_recursive(lr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                llr_layer_vec[phi].append(layer)
        for layer in range(n):
            if (phi >> layer) & 1 == 1:
                bit_layer_vec[phi].append(layer)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用 bit-reversed 译码顺序，与 B_N F^{\\otimes n} 编码一致。
    """
    llr_ch = align_llr_for_decoder(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
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
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat
