"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 + 比特倒序置换到信道顺序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{⊗n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


def polar_encode_matrix(u):
    """矩阵乘法编码（校验用）"""
    u = np.asarray(u, dtype=int)
    G = polar_generator_matrix(len(u))
    return (u @ G) % 2
