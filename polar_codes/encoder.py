"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    使用分块蝶形结构，等价于 x = u @ B_N @ F^{\otimes n}。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n_split = N
    while n_split > 1:
        half = n_split // 2
        for p in range(0, N, n_split):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        n_split //= 2
    return u[bit_reversal_permutation(N)]
