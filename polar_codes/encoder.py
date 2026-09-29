"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def _bit_reversed(i, n):
    r = 0
    for b in range(n):
        if (i >> b) & 1:
            r |= 1 << (n - 1 - b)
    return r


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，与 SCD 译码器配套）。
    输出码字按信道传输顺序排列（不做额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    n = N
    while n > 1:
        half = n // 2
        for p in range(0, N, n):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        n = half
    return u


def polar_encode_with_br(u):
    """蝶形编码后再做比特倒序（规格文档中的变体）"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def polar_encode_matrix(u):
    """GF(2) 矩阵编码，用于校验"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    return (u @ G) % 2
