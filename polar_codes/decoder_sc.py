"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index, bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _as_frozen_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return frozen_bits
    return frozen_bits.astype(bool)


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


def _upper_llr(l1, l2):
    if np.isscalar(l1):
        return float(f_operation(l1, l2))
    return f_operation(l1, l2)


def _lower_llr(l1, l2, bit):
    if bit == 0:
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _as_frozen_mask(frozen_bits)
    N = len(llr)
    llr = llr[bit_reversal_permutation(N)]
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(L, B, l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(B, l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(L, B, l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(B, l)

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=np.int32)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (i - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layers = (
            list(range(n, n - _active_bit_level(l, n), -1)) if l >= N / 2 else []
        )
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llr(P, C, phi, n, lambda_offset, llr_layer_vec):
    for l in llr_layer_vec[phi]:
        step = 1 << (l - 1)
        blocks = 1 << (n - l)
        for b in range(blocks):
            for j in range(step):
                read_idx = lambda_offset[l] + b * step + j
                write_idx = lambda_offset[l - 1] + (b // 2) * step + j
                left = P[read_idx]
                right = P[read_idx + step]
                if b % 2 == 0:
                    P[write_idx] = f_operation(left, right)
                else:
                    bit_idx = lambda_offset[l - 1] + ((b - 1) // 2) * step + j
                    P[write_idx] = g_operation(left, right, C[bit_idx])


def _update_bits(C, phi, u_bit, n, lambda_offset, bit_layer_vec):
    l = bit_reversed_index(phi, n)
    if l < (1 << (n - 1)):
        return

    u_bit = int(u_bit)
    for s in bit_layer_vec[phi]:
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                sibling = lambda_offset[s - 1] + ((j - branch_size) >> (s - 1))
                parent = lambda_offset[s - 1] + (j >> (s - 1))
                C[sibling] = int(C[parent]) ^ u_bit
                C[parent] = u_bit


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（P/C 平面数组，frozen_bits[i] 标记 u[i] 是否冻结）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _as_frozen_mask(frozen_bits)
    N = len(llr_ch)
    llr_ch = llr_ch[bit_reversal_permutation(N)]
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for phi in range(N):
        bit_idx = bit_reversed_index(phi, n)
        for s in range(n - _active_llr_level(bit_idx, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[bit_idx]:
            B[bit_idx, n] = 0
        else:
            B[bit_idx, n] = 0 if L[bit_idx, n] >= 0 else 1

        if bit_idx >= N / 2:
            for s in range(n, n - _active_bit_level(bit_idx, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(bit_idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
