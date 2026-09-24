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
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for left in range(0, N, step << 1):
            right = left + step
            u[left:left + step] ^= u[right:right + step]

    br = bit_reversal_permutation(N)
    return u[br]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    GN = (B @ G) % 2
    return (u @ GN) % 2
