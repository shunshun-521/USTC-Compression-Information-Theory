"""
极化码编码器
编码：u 上蝶形运算得到码字 x（与 SC 译码器一致的索引约定）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码，O(N log N) 蝶形结构（不在输出端做比特倒序置换）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = N
    stages = int(np.log2(N))
    for _ in range(stages):
        if n == 1:
            break
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        n = half
    return u
