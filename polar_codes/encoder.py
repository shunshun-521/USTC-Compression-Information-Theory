r"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_index(i, n):
    """单索引比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversal_index(i, n) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    等价于 x = u @ G_N，G_N = F^{\otimes n} @ B_N（GF(2)）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                u[a] = (u[a] ^ u[b]) & 1
    br = bit_reversal_permutation(N)
    return u[br].astype(np.int8)


def polar_generator_matrix(N):
    r"""返回 GF(2) 生成矩阵 G_N = F^{\otimes n} @ B_N"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (G @ B) % 2
