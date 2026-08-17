"""
极化码编码器
编码：利用蝶形递归结构实现 O(N log N) 复杂度
"""
import math
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(math.log2(N))
    return np.array([int(f'{i:0{n}b}'[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。
    采用 Arikan 极化变换的蝶形递归实现。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(math.log2(N)) + 1
    m = 1
    for _ in range(n - 1):
        for i in range(0, N, 2 * m):
            x = u[i:i + m]
            y = u[i + m:i + 2 * m]
            u[i:i + 2 * m] = np.concatenate([x ^ y, y])
        m *= 2
    return u
