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


def _update_llrs(L, B, l, n, N):
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


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr = llr[br]
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_block, offset):
        block_size = len(llr_block)
        if block_size == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_block[0] >= 0 else 1
            return

        half = block_size // 2
        llr_left = f_operation(llr_block[:half], llr_block[half:])
        decode_block(llr_left, offset)
        llr_right = g_operation(
            llr_block[:half], llr_block[half:], u_hat[offset : offset + half]
        )
        decode_block(llr_right, offset + half)

    decode_block(llr, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（比特倒序译码顺序）。"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in decode_order:
        start = n - _active_llr_level(phi, n)
        llr_layer_vec[phi] = list(range(start, n))
        bit_start = n - _active_bit_level(phi, n)
        if phi >= N // 2:
            bit_layer_vec[phi] = list(range(n, bit_start, -1))

    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr_ch = llr_ch[br]

    decode_order, _, _ = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in decode_order:
        _update_llrs(L, B, l, n, N)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        u_hat[l] = B[l, n]
        _update_bits(B, l, n, N)

    return u_hat
