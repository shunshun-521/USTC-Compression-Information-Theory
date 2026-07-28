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
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC 译码器配套）。
    采用 Arikan 核 F=[[1,1],[0,1]] 的 Kronecker 积生成矩阵。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, len(u), n):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        n = half
    return u


def build_generator_matrix(N):
    """构建生成矩阵 G_N = F^{\\otimes n}，F=[[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(F, G)
    return G % 2
