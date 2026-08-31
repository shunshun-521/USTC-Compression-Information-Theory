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
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ (B_N F^{⊗n}) mod 2
    """
    u = np.array(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    for layer in range(n):
        step = 2 ** layer
        for i in range(0, len(u), 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    rev = bit_reversal_permutation(len(u))
    return u[rev]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=int)
    rev = bit_reversal_permutation(N)
    for i, r in enumerate(rev):
        B[i, r] = 1
    G = (B @ G) % 2
    return (u @ G) % 2
