"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 PSCD 实现）
"""
import math

import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    return Lb + (1.0 - 2.0 * u_hat) * La


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


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


def precompute_sc_indices(N):
    """预计算非递归 SC 的层列表。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        start = n - _active_llr_level(i, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - _active_bit_level(i, n)
        bit_layer_vec.append(list(range(n, bit_start, -1)))
    lambda_offset = [1 << j for j in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(y, frozen_bits):
    """
    递归 SC 译码（输入为信道观测或 LLR，负值判为比特 1）。
    """
    y = np.asarray(y, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(y)
    n = int(math.log2(N)) + 1
    node_values = np.zeros(N, dtype=np.int8)

    def decode(node_y, depth, node):
        if depth == n - 1:
            if frozen_bits[node]:
                node_values[node] = 0
            else:
                # LLR 约定：L >= 0 -> 0，L < 0 -> 1
                node_values[node] = 1 if node_y[0] < 0 else 0
            return np.array([node_values[node]], dtype=np.int8)

        half = len(node_y) // 2
        L1 = node_y[:half]
        L2 = node_y[half:]
        left = f_operation(L1, L2)
        arr1 = decode(left, depth + 1, 2 * node)
        right = g_operation(L1, L2, arr1)
        arr2 = decode(right, depth + 1, 2 * node + 1)
        temp = np.concatenate([(arr1 + arr2) % 2, arr2])
        return temp

    decode(y, 0, 0)
    return node_values


def sc_decode_pscd(llr_ch, frozen_bits):
    """
    非递归 PSCD 译码（参考实现，与递归结果在部分参数下等价）。
    若输入为 LLR（正值倾向 0），内部转换为等效观测 L = -LLR 以统一判决方向。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    y = -llr_ch
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(y)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = y

    u_hat = np.zeros(N, dtype=np.int8)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(L[j - branch_size, s], L[j, s], top_bit)

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for phase in range(N):
        l = bit_reversed(phase, n)
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 1 if L[l, n] < 0 else 0
        u_hat[l] = int(B[l, n])
        update_bits(l)

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（LLR：正值倾向比特 0）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return sc_decode_recursive(llr_ch, frozen_bits)
