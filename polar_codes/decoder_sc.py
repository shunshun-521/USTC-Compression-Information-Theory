"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


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


def _prepare_llr(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    def decode_node(l_arr, idx, depth):
        if depth == 0:
            if idx in frozen_set:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if l_arr[0] >= 0 else 1
            return

        half = 1 << (depth - 1)
        l_left = f_operation(l_arr[:half], l_arr[half:])
        for i in range(half):
            decode_node(l_left[i : i + 1], idx + i, depth - 1)

        u_left = u_hat[idx : idx + half]
        l_right = g_operation(l_arr[:half], l_arr[half:], u_left)
        for i in range(half):
            decode_node(l_right[i : i + 1], idx + half + i, depth - 1)

    decode_order = [_bit_reversed(i, n) for i in range(N)]
    for start in range(0, N, N):
        pass

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr

    for i in range(N):
        l = decode_order[i]
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

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    decode_order = [_bit_reversed(i, n) for i in range(N)]
    for phi, l in enumerate(decode_order):
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec[phi] = list(range(start_s, n))

        if l >= N // 2:
            end_s = n - _active_bit_level(l, n)
            bit_layer_vec[phi] = list(range(n, end_s, -1))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = decode_order[phi]
        for s in llr_layer_vec[phi]:
            block_size = lambda_offset[s + 1]
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
            for s in bit_layer_vec[phi]:
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return u_hat
