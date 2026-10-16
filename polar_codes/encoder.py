"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，O(N log N)）。

    与 permuted SC 译码器配套：比特倒序体现在译码相位顺序中。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def build_generator_matrix(N):
    """构造极化生成矩阵 F^{\\otimes n}（未对码字做额外倒序）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(1, int(np.log2(N))):
        G = np.kron(G, F)
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码（参考实现）。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
