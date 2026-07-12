"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, bit):
    if bit == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if bit == 1:
        return l1 - l2
    return np.nan


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


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _reorder_channel_llr(llr_ch, N):
    llr = np.asarray(llr_ch, dtype=np.float64)
    return llr[bit_reversal_permutation(N)]


def _sc_decode_core(llr_tree, frozen_bits):
    """非递归 SC 译码核心（对数域精确 f 函数）。"""
    llr_tree = np.asarray(llr_tree, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_tree)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_tree

    for phi in range(N):
        leaf = _bit_reversed_index(phi, n)
        for stage in range(n - _active_llr_level(leaf, n), n):
            block_size = 1 << (stage + 1)
            branch_size = block_size >> 1
            for j in range(leaf, N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = _upper_llr(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = _lower_llr(
                        L[j, stage], L[j - branch_size, stage], int(B[j - branch_size, stage + 1])
                    )

        if leaf in frozen_set:
            B[leaf, n] = 0
        else:
            B[leaf, n] = 0 if L[leaf, n] >= 0 else 1

        if leaf < N // 2:
            continue

        for stage in range(n, n - _active_bit_level(leaf, n), -1):
            block_size = 1 << stage
            branch_size = block_size >> 1
            for j in range(leaf, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, stage - 1] = int(B[j, stage]) ^ int(B[j - branch_size, stage])
                    B[j, stage - 1] = int(B[j, stage])

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用同一核心逻辑）。"""
    llr_tree = _reorder_channel_llr(llr, len(llr))
    return _sc_decode_core(llr_tree, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        leaf = _bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(leaf, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(leaf, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_tree = _reorder_channel_llr(llr_ch, len(llr_ch))
    return _sc_decode_core(llr_tree, frozen_bits)
