"""
极化码编码器
编码：x = u * F^{⊗n}（蝶形 XOR），与置换 SC 译码器配套
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，复杂度 O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ F^{⊗n} (mod 2)
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
    return u


def polar_generator_matrix(N):
    """返回 N×N 生成矩阵 F^{⊗n}（GF(2)）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


def verify_encoder(u=None):
    """用生成矩阵验证蝶形编码正确性"""
    if u is None:
        u = np.array([1, 0, 1, 1])
    u = np.asarray(u, dtype=int)
    N = len(u)
    x = polar_encode(u)
    G = polar_generator_matrix(N)
    x_mat = (u @ G) % 2
    return np.array_equal(x, x_mat), x, x_mat
