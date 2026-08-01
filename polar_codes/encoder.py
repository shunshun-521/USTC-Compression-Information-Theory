"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    采用与 SC 译码器匹配的因子图约定：蝶形 XOR 后直接输出码字，
    比特倒序由译码器在译码顺序中处理。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n = n_split
    return u


def polar_encode_with_br(u):
    """带比特倒序置换的编码（备用）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]
