"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _normalize_frozen_bits(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return frozen_bits
    return frozen_bits.astype(bool)


def _active_llr_level(i, n):
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
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _recursive_decode(llr, frozen_bits, u_hat, idx_start, idx_end):
    length = idx_end - idx_start
    if length == 1:
        idx = idx_start
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llr[0] >= 0.0 else 1
        return

    half = length // 2
    llr_left = f_operation(llr[:half], llr[half:])
    _recursive_decode(llr_left, frozen_bits, u_hat, idx_start, idx_start + half)
    u_left = u_hat[idx_start : idx_start + half]
    llr_right = g_operation(llr[:half], llr[half:], u_left)
    _recursive_decode(llr_right, frozen_bits, u_hat, idx_start + half, idx_end)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现），返回源序列 u_hat"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _normalize_frozen_bits(frozen_bits)
    n = int(math.log2(len(llr)))
    x_hat = np.zeros(len(llr), dtype=int)
    L = np.full((len(llr), n + 1), np.nan, dtype=np.float64)
    B = np.zeros((len(llr), n + 1), dtype=int)
    L[:, 0] = llr

    for i in range(len(llr)):
        l = bit_reversed_index(i, n)
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0.0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数，返回源序列 u_hat"""
    return sc_decode_recursive(llr_ch, frozen_bits)
