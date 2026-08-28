"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
Arikan 核 F = [[1,1],[0,1]]，编码为蝶形 XOR（无输出比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(format(i, f'0{n}b')[::-1], 2)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 G = F^{⊗n} 一致，F=[[1,1],[0,1]]）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    for stage in range(n):
        n_split = N // (2 ** (stage + 1))
        step = 2 * n_split
        for p in range(0, N, step):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
    return u


def build_generator_matrix(N):
    """构造 Arikan 生成矩阵 G = F^{⊗n}"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2
