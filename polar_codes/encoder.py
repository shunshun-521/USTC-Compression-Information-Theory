"""
极化码编码器
编码：蝶形结构 O(N log N)，与标准极化码因子图一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（分块蝶形 XOR，与标准 SCD 译码器配套）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    stages = int(np.log2(N))
    for _ in range(stages):
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                left = p + k
                u[left] ^= u[left + n_split]
        n = n_split
    return u


def build_generator_matrix(N):
    """构建生成矩阵 G_N，满足 x = u @ G_N（用于验证）"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        u = np.zeros(N, dtype=int)
        u[i] = 1
        G[i] = polar_encode(u)
    return G
