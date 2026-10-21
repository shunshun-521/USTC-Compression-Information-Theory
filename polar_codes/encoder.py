"""
极化码编码器
编码：Arikan 极化变换（蝶形 XOR），O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed_int(i, n) for i in range(N)], dtype=np.int64)


def bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 Permuted SCD 译码器配套）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half

    return u.astype(np.int8)


def polar_encode_matrix(u):
    """GF(2) 矩阵乘法（G_N = B_N F^{\\otimes n}，用于校验）"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=np.int8)
    rev = bit_reversal_permutation(N)
    for i in range(N):
        B[i, rev[i]] = 1
    GN = (B @ G) % 2
    return (u @ GN) % 2
