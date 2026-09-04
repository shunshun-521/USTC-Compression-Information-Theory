"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于分层 LLR/比特数组）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.where(
        La * Lb >= 0,
        np.minimum(np.abs(La), np.abs(Lb)),
        -np.minimum(np.abs(La), np.abs(Lb)),
    )


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(i, n):
    """单索引比特倒序。"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def _active_llr_level(i, n):
    """llr 更新起始层（第一个 1 的位置，从 MSB 计数）。"""
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
    """比特回传起始层（第一个 0 的位置，从 MSB 计数）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _inverse_bit_reversal(N):
    """比特倒序置换的逆映射。"""
    brp = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    for i, j in enumerate(brp):
        inv[j] = i
    return inv


def _sc_decode_core(llr, frozen_bits):
    """SC 译码核心（返回自然序源比特估计）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    inv_brp = _inverse_bit_reversal(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n)
        src_idx = inv_brp[l]
        if frozen_bits[src_idx]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    brp = bit_reversal_permutation(N)
    return B[:, n].astype(int)[brp]


def _update_llrs(L, B, l, n):
    """自底向上更新 LLR 树。"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    """自顶向下更新部分和比特树。"""
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（分层数组实现，与 sc_decode 等价）。
    """
    return _sc_decode_core(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    return _sc_decode_core(llr_ch, frozen_bits)
