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
    极化码编码：蝶形结构 + 比特倒序置换。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(math.log2(N))

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]

    brp = bit_reversal_permutation(N)
    return u[brp]
