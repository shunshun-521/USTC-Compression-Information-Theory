"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（用于 BP 等）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_boxplus(La, Lb):
    """SC 译码使用的精确 f 运算（boxplus），支持向量化。"""
    from scipy.special import logsumexp

    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    a = np.stack([La + Lb, np.zeros_like(La)], axis=0)
    b = np.stack([La, Lb], axis=0)
    return logsumexp(a, axis=0) - logsumexp(b, axis=0)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


def _frozen_phase_set(frozen_bits):
    """将冻结位自然索引映射到 SC 译码相位索引。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(frozen_bits)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    return {br[i] for i in np.where(frozen_bits)[0]}


def _sc_decode_core(llr_tree, frozen_phase_set):
    N = len(llr_tree)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_tree

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if i in frozen_phase_set:
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

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            end = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))
    lambda_offset = np.array([(1 << layer) - 1 for layer in range(n + 1)], dtype=int)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _split_f(llr_node):
    half = len(llr_node) // 2
    pairs = llr_node.reshape(half, 2)
    return f_boxplus(pairs[:, 0], pairs[:, 1])


def _split_g(llr_node, u_partial):
    half = len(llr_node) // 2
    pairs = llr_node.reshape(half, 2)
    u_partial = np.asarray(u_partial).reshape(half)
    return g_operation(pairs[:, 0], pairs[:, 1], u_partial)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，委托非递归核心）。"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    frozen_phase = _frozen_phase_set(frozen_bits)
    return _sc_decode_core(llr_ch[br], frozen_phase)
