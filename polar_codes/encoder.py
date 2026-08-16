"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：u[j] ^= u[j + step]，共 log2(N) 层；
    最后做比特倒序置换 x[i] = u[bit_reversed(i)]。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    stage_n = N

    for _ in range(n):
        n_split = stage_n // 2
        for p in range(0, N, stage_n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        stage_n = n_split

    br = bit_reversal_permutation(N)
    return u[br]
