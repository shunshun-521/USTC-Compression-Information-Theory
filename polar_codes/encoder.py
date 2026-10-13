"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)
    return rev


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
    assert 2 ** n == N

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x.astype(int)


def build_generator_matrix(N):
    """GF(2) 生成矩阵 G_N = B_N F^{\\otimes n}，用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    G = G[rev, :]
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码（校验用）。"""
    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
