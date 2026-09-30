"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u @ (B_N F^{\\otimes n})
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for step in range(n):
        inc = 2 ** (step + 1)
        half = 2 ** step
        for i in range(0, N, inc):
            for j in range(half):
                u[i + j] ^= u[i + j + half]
    br = bit_reversal_permutation(N)
    return u[br]
