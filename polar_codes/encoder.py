"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    n = len(u)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    G = np.array([[1]], dtype=int)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F) % 2
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return (u @ G) % 2
