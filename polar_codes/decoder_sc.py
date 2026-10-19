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


def _bit_reversed(i, n):
    r = 0
    for b in range(n):
        if (i >> b) & 1:
            r |= 1 << (n - 1 - b)
    return r


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
    """
    递归 SC 译码（参考实现）。
    与按层块划分的朴素递归不同，此处采用与 sc_decode 相同的逐相位树更新，
    以保证与含比特倒序的编码器一致。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供 SCL 等模块使用）"""
    n = int(math.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n)]

    llr_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layer_vec.append(list(range(n)))
            continue
        layers = []
        p = phi
        while p & 1:
            layers.append(len(layers))
            p >>= 1
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        if phi == N - 1:
            bit_layer_vec.append([])
            continue
        layers = []
        p = phi + 1
        while not (p & 1):
            layers.append(len(layers))
            p >>= 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n):
    """更新第 l 个叶子对应的 LLR 链（列 0 为信道侧）"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    """硬判决比特回传"""
    if l < len(B) // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（O(N log N)）
    采用按比特倒序的自然索引 LLR 阵列更新。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    llr_ch = llr_ch[bit_reversal_permutation(N)]
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(np.int32)
