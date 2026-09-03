"""
极化码编码器
编码：蝶形结构 O(N log N)，与 G_N = F^{\\otimes n} 一致（F=[[1,1],[0,1]]）
"""
import numpy as np


def bit_reversal_index(i, n):
    """对标量索引 i 做 n 位比特倒序。"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversal_index(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无输出比特倒序）。
    与 u @ G_N (mod 2) 一致，G_N = F^{\\otimes n}, F=[[1,1],[0,1]]。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def polar_generator_matrix(N):
    """生成极化码生成矩阵 G_N = F^{\\otimes n}。"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(F, G)
    return G


def polar_encode_matrix(u):
    """基于生成矩阵的编码（参考实现）：x = G_N @ u (mod 2)。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    return (polar_generator_matrix(N) @ u) % 2
