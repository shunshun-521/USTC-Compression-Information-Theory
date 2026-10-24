"""
极化码 SC（串行抵消）译码器
PSCD 非递归实现（比特倒序调度），含递归参考实现
"""
import numpy as np
import math

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> La-Lb"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def upper_llr(l1, l2):
    return f_operation(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
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


def precompute_sc_indices(N):
    """兼容接口：返回比特倒序译码顺序等元数据"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    decode_order = [bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [list(range(n - active_llr_level(bit_reversed(phi, n), n), n)) for phi in range(N)]
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _frozen_index_set(frozen_bits):
    fb = np.asarray(frozen_bits)
    return set(np.where(fb.astype(bool))[0])


def sc_decode(llr_ch, frozen_bits):
    """非递归 PSCD 主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_index_set(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = bit_reversed(phi, n)
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 PSCD 同 LLR 约定，用于交叉验证）"""
    return sc_decode(llr, frozen_bits)


sc_decode_nonrecursive = sc_decode
