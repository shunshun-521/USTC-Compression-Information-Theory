"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    利用蝶形结构实现 x = u * G_N（G_N 为极化核的 Kronecker 幂，块递归形式），
    复杂度 O(N log N)。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 2 ** layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    return u


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N（用于验证）"""
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        z = np.zeros_like(G, dtype=int)
        G = np.block([[G, z], [G, G]]) % 2
    return G
