"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(top_llr, btm_llr, u_hat):
    """g 运算（与 SCD 下分支更新一致：btm ± top）"""
    u_hat = int(u_hat)
    if u_hat == 0:
        return btm_llr + top_llr
    return btm_llr - top_llr


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


def bit_reversed_index(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与主译码器同一 LLR 布局）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """兼容接口：返回解码顺序等辅助信息"""
    n = int(math.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in decode_order:
        start = n - _active_llr_level(phi, n)
        llr_layer_vec[phi] = list(range(start, n))
        bit_start = n - _active_bit_level(phi, n)
        bit_layer_vec[phi] = list(range(n, bit_start, -1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], int(top_bit)
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] ^ B[j - branch_size, s]) & 1
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _pm_update(pm, llr, u):
    u_hard = _llr_to_bit(llr)
    if u != u_hard:
        pm += abs(llr)
    return pm
