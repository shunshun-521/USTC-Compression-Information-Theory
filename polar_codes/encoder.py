"""
极化码编码器
编码：x = u * F^⊗n，蝶形结构 O(N log N)
（与 SC 译码器采用相同的自然序比特索引约定）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序（与译码阶段顺序一致）。"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = N
    while n > 1:
        half = n // 2
        for p in range(0, N, n):
            for k in range(half):
                l = p + k
                u[l] = (u[l] ^ u[l + half]) & 1
        n = half
    return u


def polar_encode_matrix(u):
    """矩阵法编码（用于校验）：x = u @ F^⊗n mod 2。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F) % 2
    return (u @ F_n) % 2
