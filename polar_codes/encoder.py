"""
极化码编码器
编码：x = u * F^{\\otimes n}（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Kronecker F^{\\otimes n} 蝶形，与 SC/SCL 译码器一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1
    return u


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    F_n = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    return F_n


def polar_encode_with_reversal(u):
    """带比特倒序置换的编码 x = u * B_N * F^{\\otimes n}"""
    x = polar_encode(u)
    perm = bit_reversal_permutation(len(x))
    return x[perm]


def polar_encode_matrix(u):
    """矩阵法编码 u @ G_N"""
    u = np.asarray(u, dtype=np.int8)
    return (u @ build_generator_matrix(len(u))) % 2
