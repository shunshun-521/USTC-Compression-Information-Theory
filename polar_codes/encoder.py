"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def bit_reversed(i, n):
    """单索引比特倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    Arikan 蝶形：u[i] ^= u[i+step]，最后对比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = F.copy()
    for _ in range(n - 1):
        Fn = np.kron(Fn, F)
    rev = bit_reversal_permutation(N)
    Gn = (np.eye(N, dtype=np.int8)[rev] @ Fn) % 2
    return (u @ Gn) % 2
