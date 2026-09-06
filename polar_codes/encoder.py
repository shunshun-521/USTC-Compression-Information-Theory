"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8).copy()
    n = len(u)
    if n == 0 or (n & (n - 1)):
        raise ValueError("u length must be a power of 2")

    step = n
    while step > 1:
        step //= 2
        for base in range(0, n, 2 * step):
            for j in range(step):
                u[base + j] ^= u[base + j + step]

    return u[bit_reversal_permutation(n)]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}，用于验证"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G[bit_reversal_permutation(N), :]
