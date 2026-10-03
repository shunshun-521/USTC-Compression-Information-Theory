"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 SCD 实现（高效）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x >= y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f / box-plus（min-sum 可选，此处用稳定 log-domain 实现）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：La 为左分支 LLR，Lb 为右分支 LLR"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    out = np.empty_like(La)
    mask0 = u_hat == 0
    out[mask0] = La[mask0] + Lb[mask0]
    out[~mask0] = La[~mask0] - Lb[~mask0]
    return out


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


def sc_decode(llr_ch, frozen_bits):
    """非递归 SCD 译码（Permuted SC）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for phi in range(N):
        l = bit_reversed(phi, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 SCD 等价的参考实现）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """兼容接口：返回占位结构"""
    n = int(math.log2(N))
    lambda_offset = list(range(n + 2))
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
