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
    极化码编码（含比特倒序置换）。

    蝶形 XOR：u[i] ^= u[i+step]，与 G_N = B_N F^{\\otimes n} 一致。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step //= 2

    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    F_n = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        F_n = np.kron(F_n, F)
    br = bit_reversal_permutation(N)
    G = F_n[br, :]
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码（验证用）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
