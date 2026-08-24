"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode_butterfly(u):
    """
    蝶形递归编码（不含比特倒序），用于与矩阵乘法对照。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    return u


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
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{⊗n}（GF(2)），用于验证"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = F.copy()
    for _ in range(n - 1):
        Fn = np.kron(Fn, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ Fn) % 2


def polar_encode_matrix(u):
    """矩阵乘法编码 u @ G_N"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
