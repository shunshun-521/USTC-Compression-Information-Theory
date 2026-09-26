"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    v = np.array(u, dtype=np.int8, copy=True)
    N = len(v)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                v[j] ^= v[j + step]
        step <<= 1
    brp = bit_reversal_permutation(N)
    return v[brp].astype(int)


def polar_encode_matrix(N):
    """生成 G_N 用于验证（u @ G = polar_encode(u)）。"""
    u = np.eye(N, dtype=int)
    return np.vstack([polar_encode(u[i]) for i in range(N)])
