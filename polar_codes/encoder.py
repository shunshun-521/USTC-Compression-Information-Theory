"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    return ((indices >> np.arange(n)) & 1).sum(axis=1) if False else _bit_rev_indices(N)


def _bit_rev_indices(N):
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r = (r << 1) | ((i >> b) & 1)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：x = u * G_N，G_N = F^⊗n。

    利用蝶形结构实现 O(N log N) 复杂度，与标准 SC 译码器配套。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            u[base:base + half] ^= u[base + half:base + n]
        n = half
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
    print("matrix encode =", polar_encode_matrix(u))
