"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_generate_matrix(N):
    """生成极化码生成矩阵（Kronecker 积，与 polar_encode 一致）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码（Kronecker 蝶形结构，与 SC/SCL 译码器配套）。
    与 SC/SCL 译码器配套使用。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(1, n + 1):
        block = 1 << layer
        half = block // 2
        for block_start in range(0, N, block):
            u[block_start:block_start + half] ^= u[block_start + half:block_start + block]

    return u
