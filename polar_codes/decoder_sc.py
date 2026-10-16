"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.maximum(x, y) + np.log1p(np.exp(-np.abs(x - y)))


def f_operation(l1, l2):
    """f 运算（对数域 box-plus，用于 SC，支持向量化）。"""
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def g_operation(l1, l2, u_hat):
    """g 运算（对数域，支持向量化 u_hat）。"""
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.int8)
    return np.where(u_hat == 0, l1 + l2, l1 - l2)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（与 permuted SC 相位一致）。"""
    n = int(math.log2(N))
    lambda_offset = [0]
    for i in range(n):
        lambda_offset.append(lambda_offset[-1] + (1 << i))

    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=np.int8)
        return np.array([0 if llr[0] >= 0 else 1], dtype=np.int8)

    half = N // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[:half], llr[half:], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def sc_decode_fast(llr_ch, frozen_bits):
    """别名：与非递归主实现相同。"""
    return sc_decode(llr_ch, frozen_bits)
