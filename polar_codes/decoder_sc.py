"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block = 1 << (s + 1)
        branch = block // 2
        for j in range(l, N, block):
            if j % block < branch:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
            else:
                top_bit = B[j - branch, s + 1]
                L[j, s + 1] = g_operation(L[j - branch, s], L[j, s], top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block = 1 << s
        branch = block // 2
        for j in range(l, -1, -block):
            if j % block >= branch:
                B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr_int = llr[rev]
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        if depth == 0:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = len(llr_node) // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, depth - 1, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, depth - 1, bit_offset + half)

    decode_node(llr_int, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        if phi < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch[rev]

    u_hat = np.zeros(N, dtype=int)
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        _update_bits(B, l, n, N)

    return u_hat
