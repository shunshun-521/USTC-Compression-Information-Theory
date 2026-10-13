"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 rev，满足 out[i] = in[rev[i]]"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    bits = (idx[:, None] >> np.arange(n)) & 1
    rev = (bits * (1 << np.arange(n - 1, -1, -1))).sum(axis=1).astype(int)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            left = u[i : i + step]
            right = u[i + step : i + 2 * step]
            u[i : i + step] = (left ^ right) & 1
        step <<= 1
    br = bit_reversal_permutation(N)
    x = u[br]
    return x
