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
    u: 源序列（信息位 + 冻结位），长度 N
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """显式生成矩阵 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, br[i]] = 1
    return (B @ G) % 2
