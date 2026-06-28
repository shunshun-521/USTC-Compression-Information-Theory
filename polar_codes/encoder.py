"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR 结构，与 G_N = F^n 一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    if 2 ** n != len(u):
        raise ValueError("Length of u must be a power of 2")

    for s in range(n):
        step = 1 << s
        for i in range(0, len(u), 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
    return u


def polar_encode_with_reversal(u):
    """带比特倒序置换的编码（部分文献约定）。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]


def build_generator_matrix(N):
    """构造生成矩阵 G_N = F^n（行向量编码 u @ G）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G
