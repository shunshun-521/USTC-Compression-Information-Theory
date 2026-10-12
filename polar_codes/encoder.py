"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int64)
    rev = (bits * (2 ** np.arange(n)[::-1])).sum(axis=1)
    return rev.astype(np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
    br = bit_reversal_permutation(N)
    return u[br].astype(np.int8)
