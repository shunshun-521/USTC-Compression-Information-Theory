"""
极化码编码器
编码：蝶形 XOR（与 3GPP / 常用参考实现一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if N != 2 ** n:
        raise ValueError("N must be a power of 2")
    return np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if N != 2 ** n:
        raise ValueError("N must be a power of 2")
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        block = half
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵法编码，用于校验。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F) % 2
    return (u @ G) % 2
