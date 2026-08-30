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
        result = 0
        for b in range(n):
            if i & (1 << b):
                result |= 1 << (n - 1 - b)
        rev[i] = result
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 核 F=[[1,1],[0,1]]，蝶形 u[i] ^= u[i+split]）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n = n_split
    return u


def polar_encode_matrix(N):
    """生成编码矩阵 G_enc = F^{\\otimes n,T}，满足 x = u @ G_enc"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G.T
