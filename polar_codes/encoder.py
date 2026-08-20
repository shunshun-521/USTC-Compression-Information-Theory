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

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.array(u, dtype=int).copy()
    n = len(u)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Length of u must be a power of 2")

    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1

    br = bit_reversal_permutation(n)
    return u[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}，用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[:, br]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（校验用）。"""
    u = np.array(u, dtype=int)
    G = build_generator_matrix(len(u))
    return (u @ G) % 2
