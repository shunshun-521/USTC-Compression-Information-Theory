"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """LLR 更新起始层（首个 1 的位置，自高位计）"""
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
    """比特回传起始层（首个 0 的位置，自高位计）"""
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
    """min-sum 近似的 f 运算（向量化）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def upper_llr(l1, l2):
    """f 运算（标量，min-sum）"""
    return float(f_operation(l1, l2))


def lower_llr(l1, l2, b):
    """g 运算（标量）"""
    return float(g_operation(l1, l2, b))


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted / natural-order 实现，参考 Vangala et al.）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    if N & (N - 1):
        raise ValueError("N must be power of 2")

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = lower_llr(L[j - branch_size, s], L[j, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        if l < N / 2:
            continue
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat


def _partial_encode(u):
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    m = int(math.log2(N))
    for layer in range(m):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
    return u


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，g 分支使用子块编码后的比特）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, idx, length):
        if length == 1:
            pos = idx
            if frozen_bits[pos]:
                u_hat[pos] = 0
            else:
                u_hat[pos] = 0 if llr_node[0] >= 0 else 1
            return
        half = length // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, idx, half)
        u_left = u_hat[idx:idx + half]
        v = _partial_encode(u_left)
        llr_right = g_operation(llr_node[:half], llr_node[half:], v)
        decode_node(llr_right, idx + half, half)

    decode_node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（与按位更新层对应）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layers = list(range(n - active_llr_level(l, n), n))
        bit_layers = list(range(n, n - active_bit_level(l, n), -1)) if l >= N / 2 else []
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
