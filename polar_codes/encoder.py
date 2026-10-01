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
    极化码编码（stage 蝶形，与 SC 因子图一致的索引顺序，末尾不做 B_N 置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, step << 1):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵形式编码（用于验证）：x = u @ G_N mod 2"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i, j in enumerate(br):
        B[i, j] = 1
    G_N = (u @ (B @ G)) % 2
    return G_N.astype(int)
