"""
极化码编码器
编码：蝶形结构实现 u -> x = u F^{\\otimes n}（O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            r |= ((i >> b) & 1) << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，无额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u
