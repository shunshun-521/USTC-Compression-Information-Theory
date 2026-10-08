"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _logdomain_sum(a, b):
    return float(np.logaddexp(a, b))


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 复用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（供 SCL/BP 复用）。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，min-sum）。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    llr = llr[bit_reversal_permutation(N)]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n_len = len(llr_node)
        if n_len == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] else (0 if llr_node[0] >= 0 else 1)
            return

        half = n_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """SCL 仍使用的层向量（与 Permuted SCD 兼容的近似调度）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    llr_ch = llr_ch[bit_reversal_permutation(N)]
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
