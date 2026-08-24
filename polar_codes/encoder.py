"""
极化码编码器
编码：x = u * G_N^T，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，O(N log N)）。

    对 u 执行 log2(N) 层蝶形：u[i] ^= u[i+step]
    等价于 x = u @ G_N^T，G_N = F^{\\otimes n}，F=[[1,1],[0,1]]
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    size = N
    while size > 1:
        half = size // 2
        for p in range(0, N, size):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        size = half
    return u


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N = F^{\\otimes n}（Arikan 标准核）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(F, G)
    return G % 2
