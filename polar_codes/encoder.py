"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。
    与 permuted SC 译码器配套：x = u * F^{\\otimes n}（不做比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for j in range(half):
                idx = start + j
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_generator_matrix(N):
    """构造生成矩阵 G_N = F^{\\otimes n}（用于验证）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    m = int(np.log2(N)) - 1
    for _ in range(m):
        G = np.kron(G, F)
    return G % 2
