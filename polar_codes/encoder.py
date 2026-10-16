"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation_fast(N):
    n = int(np.log2(N))
    return ((np.arange(N)[:, None] >> np.arange(n)) & 1).dot(1 << np.arange(n - 1, -1, -1))


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return bit_reversal_permutation_fast(N)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    br = bit_reversal_permutation_fast(N)
    x = u[br]
    return x.astype(np.int8)
