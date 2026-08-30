"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversed(i, n):
    """单索引比特倒序（与 polarcodes 一致）"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，无额外比特倒序）。
    与 Permuted SCD 译码器配套使用。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def build_generator_matrix(N):
    """构建极化码生成矩阵（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(1, int(np.log2(N))):
        G = np.kron(G, F)
    return G
