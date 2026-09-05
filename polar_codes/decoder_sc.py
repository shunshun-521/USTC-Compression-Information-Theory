"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    f 运算（log-domain boxplus，与 min-sum 兼容的高 SNR 行为更好）。
    向量化 wrapper，内部逐元素调用 boxplus。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(_boxplus, otypes=[float])(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _lower_llr(btm, top, b):
    """下分支 LLR 更新（btm=当前, top=配对分支）"""
    if b == 0:
        if np.isinf(btm) or np.isinf(top):
            return np.inf
        return btm + top
    return btm - top


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _boxplus(l1, l2):
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    if np.isinf(l1):
        return l2
    if np.isinf(l2):
        return l1
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（自然顺序，log-domain f）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算 Permuted SCD 辅助信息。
    返回 bit-reversed 相位顺序及层索引向量。
    """
    n = int(math.log2(N))
    phase_order = [bit_reversal_permutation(N)[i] for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in phase_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in phase_order]
    return phase_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（O(N log N)）。
    信道 LLR 按自然顺序输入，内部按 bit-reversed 相位顺序译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    phase_order, _, _ = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for l in phase_order:
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block = 1 << (s + 1)
            branch = block >> 1
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = _boxplus(L[j, s], L[j + branch, s])
                else:
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
            u_hat[l] = B[l, n]

        if l >= N // 2:
            start_s = n - _active_bit_level(l, n)
            for s in range(n, start_s, -1):
                block = 1 << s
                branch = block >> 1
                for j in range(l, -1, -block):
                    if j % block >= branch:
                        B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                        B[j, s - 1] = B[j, s]

    return u_hat
