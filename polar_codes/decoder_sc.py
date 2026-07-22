"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    对数域 f 运算（boxplus），支持向量化。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    for idx in range(La.size):
        a = float(La.ravel()[idx])
        b = float(Lb.ravel()[idx])
        if np.isinf(a) and not np.isinf(b):
            out.ravel()[idx] = b
        elif not np.isinf(a) and np.isinf(b):
            out.ravel()[idx] = a
        elif np.isinf(a) and np.isinf(b):
            out.ravel()[idx] = np.inf
        else:
            out.ravel()[idx] = logdomain_sum(a + b, 0.0) - logdomain_sum(a, b)
    return out.reshape(La.shape)


def g_operation(La, Lb, u_hat):
    """
    对数域 g 运算。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    out = np.empty_like(La)
    for idx in range(La.size):
        a = float(La.ravel()[idx])
        b = float(Lb.ravel()[idx])
        bit = int(u_hat.ravel()[idx])
        if bit == 0:
            if np.isinf(a) or np.isinf(b):
                out.ravel()[idx] = np.inf
            else:
                out.ravel()[idx] = a + b
        else:
            out.ravel()[idx] = a - b
    return out.reshape(La.shape)


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


def _prepare_channel_llrs(llr_ch):
    """编码器输出含比特倒序，将信道 LLR 映射到译码树顺序"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = _prepare_channel_llrs(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, offset, length):
        if length == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = length // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, offset, half)
        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, offset + half, half)

    decode_node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l_br = bit_reversed_index(phi, n)
        start = n - active_llr_level(l_br, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - active_bit_level(l_br, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = _prepare_channel_llrs(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    for l in [bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s],
                        L[j - branch_size, s],
                        B[j - branch_size, s + 1],
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
