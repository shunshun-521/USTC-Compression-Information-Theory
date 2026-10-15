"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：u[i] ^= u[i+step]，step = 1,2,4,...
    输出按比特倒序置换索引重排。
    """
    u = np.array(u, dtype=np.int8).copy()
    n_len = len(u)
    step = 1
    while step < n_len:
        for i in range(0, n_len, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step <<= 1
    br = bit_reversal_permutation(n_len)
    x = u[br].astype(np.int8)
    return x


def generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（GF(2)），用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


def polar_encode_matrix(u):
    """矩阵乘法编码（校验用）。"""
    u = np.array(u, dtype=np.int8)
    N = len(u)
    G = generator_matrix(N)
    return (u @ G) % 2
