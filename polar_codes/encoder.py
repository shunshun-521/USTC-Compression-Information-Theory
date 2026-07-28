"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for bit in range(n):
            if (i >> bit) & 1:
                rev[i] |= 1 << (n - 1 - bit)
    return rev


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for bit in range(n):
        if (i >> bit) & 1:
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，O(N log N)）。
    等价于 x = u @ F^{\otimes n} (mod 2)。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    n_split = N
    for _ in range(n):
        n_split //= 2
        for p in range(0, N, 2 * n_split):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
