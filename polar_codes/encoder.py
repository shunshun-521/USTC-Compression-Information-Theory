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
    极化码编码（蝶形结构，等价于 u 乘以 F 的 n 次 Kronecker 积）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = u.shape[0]
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            block = u[i : i + 2 * step]
            left = block[:step]
            right = block[step:]
            block[:step] = (left ^ right) & 1
            u[i : i + 2 * step] = block
        step <<= 1
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵形式验证：u @ G_N"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
