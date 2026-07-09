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
    极化码编码（蝶形 XOR，与 PSC 译码器匹配）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = N
    while step > 1:
        half = step // 2
        for p in range(0, N, step):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        step = half
    return u


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N = B_N F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]
