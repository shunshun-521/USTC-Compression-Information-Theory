"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u * G_N（蝶形 O(N log N) 实现）。
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    k = N // 2
    while k > 0:
        for j in range(0, N, 2 * k):
            for i in range(k):
                x[j + i] ^= x[k + j + i]
        k >>= 1
    return x


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于 BP 早停校验）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
