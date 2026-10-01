"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode_butterfly(u):
    """蝶形编码（不含 B_N）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码（蝶形结构；B_N 由 Permuted SCD 译码端等效处理）。
    """
    return polar_encode_butterfly(u)


def build_generator_matrix(N):
    """构造 F^{\\otimes n}（与蝶形编码一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


def polar_encode_matrix(u):
    """矩阵乘法编码（参考）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
