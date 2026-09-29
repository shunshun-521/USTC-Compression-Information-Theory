"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in idx], dtype=int)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x[j] = v[bit_reverse(j)]，其中 v 为蝶形变换后的向量（等价于 u 乘以 B_N 与 F 的 n 次 Kronecker 积）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1
    br = bit_reversal_permutation(N)
    x = u[br]
    return x


def polar_encode_matrix(u):
    """GF(2) 生成矩阵编码，用于 BP 早停校验"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i in range(N):
        B[br[i], i] = 1
    GN = (B @ G) % 2
    return (u @ GN) % 2
