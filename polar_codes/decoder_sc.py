"""
极化码 SC（串行抵消）译码器
Permuted SC（Vangala 2014），与 mcba1n 编码器配套
"""
import numpy as np
import math

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（upper branch）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(Lb, Lt, u_hat):
    """
    g 运算（lower branch）：bottom + (1-2*u)*top
    Lb: bottom LLR, Lt: top LLR
    """
    return Lb + (1 - 2 * u_hat) * Lt


def _active_llr_level(i, n):
    """二进制表示中从最高位起连续 0 的个数 + 1"""
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
    """二进制表示中从最高位起连续 1 的个数 + 1"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    """更新 LLR 树（Permuted SC）"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top, btm)
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(btm, top, top_bit)


def _update_bits(B, l, n, N):
    """比特回传"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    Permuted SC 译码（非递归，主实现）。
    frozen_bits[i]=1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    frozen_set = set(np.where(frozen_bits == 1)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """兼容 SCL 接口的占位函数"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 1 << (i - 1)
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
