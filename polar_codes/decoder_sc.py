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
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u = np.asarray(u_hat)
    return (1.0 - 2.0 * u) * La + Lb


def _active_llr_level(i, n):
    """返回 LLR 更新起始层。"""
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
    """返回比特回传起始层。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    """非递归更新第 l 个比特对应的 LLR。"""
    N = L.shape[0]
    start = n - _active_llr_level(l, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if (j % block_size) < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    """非递归更新部分和。"""
    N = B.shape[0]
    if l < N // 2:
        return
    end = n - _active_bit_level(l, n)
    for s in range(n, end, -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if (j % block_size) >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（按比特倒序处理）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    def decode_node(llr_node, bit_offset):
        m = len(llr_node)
        if m == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)

        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_node(llr_left, bit_offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_node(llr_right, bit_offset + half)
        return np.concatenate([u_left, u_right])

    u_tree = decode_node(llr[br], 0)
    u_hat = np.empty(N, dtype=int)
    u_hat[br] = u_tree
    return u_hat


def precompute_sc_indices(N):
    """
    预计算辅助向量（供 SCL 复用）。
    """
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    decode_order = [int(br[i]) for i in range(N)]
    lambda_offset = np.array([1 << layer for layer in range(n + 1)], dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l >= N // 2:
            end = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    _, _, _, decode_order = _get_sc_cache(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for l in decode_order:
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)
