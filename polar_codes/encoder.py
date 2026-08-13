"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversed_int(x, n):
    """对 n 位整数 x 做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed_int(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码蝶形编码（与标准 polarLib 实现一致）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for ll in range(1, n + 1):
        n_iter = 1 << ll
        half = n_iter // 2
        for kk in range(N // n_iter):
            start = kk * n_iter
            u[start:start + half] ^= u[start + half:start + n_iter]
    return u


def polar_encode_matrix(u):
    """小码长验证：G_N = B_N F^{\\otimes n}，x = u @ G_N"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    G = G[rev, :]
    return (u @ G) % 2
