"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码：x = u * G_N（蝶形 + 可选比特倒序，O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step <<= 1

    if apply_bit_reversal:
        rev = bit_reversal_permutation(N)
        return u[rev]
    return u


def polar_encode_matrix(u):
    """矩阵乘法编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=np.int8)
    rev = bit_reversal_permutation(N)
    for i in range(N):
        B[i, rev[i]] = 1
    GN = (G @ B) % 2
    return (u @ GN) % 2
