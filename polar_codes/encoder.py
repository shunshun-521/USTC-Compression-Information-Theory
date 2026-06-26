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
    极化码编码：x = u @ G_N，其中 G_N = B_N @ F^{\\otimes n}。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：
        1. 对输入做比特倒序置换 u = u[B_N]
        2. 蝶形 XOR：每层 u[i] ^= u[i + step]
        3. 直接返回（输出不再做比特倒序）
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    brp = bit_reversal_permutation(N)
    u = u[brp]
    for layer in range(n):
        step = 2 ** (n - layer - 1)
        for j in range(2 ** layer):
            base = 2 * step * j
            for i in range(step):
                u[base + i] ^= u[base + i + step]
    return u


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}，用于验证编码器。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.block([[G, np.zeros_like(G)], [G, G]])
    brp = bit_reversal_permutation(N)
    return G[brp, :]


def polar_encode_matrix(u):
    """通过生成矩阵编码（验证用）。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
