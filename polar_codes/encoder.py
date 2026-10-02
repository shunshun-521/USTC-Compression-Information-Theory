"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    x = np.array(u, dtype=np.int8, copy=True)
    n = len(x)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            x[i:i + step] ^= x[i + step:i + 2 * step]
        step <<= 1
    br = bit_reversal_permutation(n)
    return x[br].astype(int)
