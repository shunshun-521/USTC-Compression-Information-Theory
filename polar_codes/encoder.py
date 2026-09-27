"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    assert 2 ** n == N
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


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
    while step < N:
        for i in range(0, N - step, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（GF(2)）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


def polar_encode_matrix(u):
    """矩阵乘法编码（用于校验）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
