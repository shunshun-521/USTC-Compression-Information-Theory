"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversed(x, n):
    """单个索引的比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC 译码器匹配）。
    等价于 u 乘以 F 的 n 次 Kronecker 积（输出倒序由译码器遍历顺序处理）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = N
    for _ in range(int(np.log2(N))):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u
