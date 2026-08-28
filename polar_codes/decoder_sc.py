"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，mcba1n 风格）
"""
import math

import numpy as np


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
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(a, b):
    """log(exp(a) + exp(b))，数值稳定实现。"""
    if np.isscalar(a):
        if a == -np.inf and b == -np.inf:
            return -np.inf
        if a == np.inf or b == np.inf:
            return np.inf
        m = max(a, b)
        return m + np.log1p(np.exp(-abs(a - b)))
    out = np.maximum(a, b)
    return out + np.log1p(np.exp(-np.abs(a - b)))


def upper_llr(l1, l2):
    """f 运算（精确 box-plus）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, bit):
    """g 运算。"""
    if bit == 0:
        return l1 + l2
    return l1 - l2


def hard_decision(llr):
    """LLR 硬判决：正号倾向 0。"""
    return 0 if llr >= 0 else 1


def bit_reversed(i, n):
    """n 位比特倒序。"""
    result = 0
    for _ in range(n):
        result = (result << 1) | (i & 1)
        i >>= 1
    return result


def active_llr_level(i, n):
    """找到 i 的二进制展开中第一个 1 的位置（从高位计）。"""
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
    """找到 i 的二进制展开中第一个 0 的位置（从高位计）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, depth, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i:i + 1], depth - 1, bit_offset + i)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], depth - 1, bit_offset + half + i)
        return None

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（比特倒序译码顺序）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    frozen_set = set(np.where(frozen_bits)[0])
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    def update_llrs(l):
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

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for phi in range(N):
        l = bit_reversed(phi, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = hard_decision(L[l, n])
        u_hat[l] = int(B[l, n])
        update_bits(l)

    return u_hat
