"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    采用 Arikan 蝶形：对每层的左半部分执行 u[l] ^= u[l + n_split]。
    比特倒序置换在 SC/SCL 译码器的译码顺序中体现。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block_size = N
    for _ in range(n):
        if block_size == 1:
            break
        half = block_size // 2
        for base in range(0, N, block_size):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block_size = half
    return u
