"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：从大到小分区（与 mcba1n polar_encode 一致）+ 输出比特倒序。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    n_split = N

    for _ in range(n):
        if n_split == 1:
            break
        half = n_split // 2
        for p in range(0, N, n_split):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        n_split = half

    br = bit_reversal_permutation(N)
    x = u[br]
    return x


def polar_encode_matrix(u):
    """矩阵法编码（用于验证），GF(2) 上 x = u @ G_N"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))

    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)

    br = bit_reversal_permutation(N)
    BN = np.eye(N, dtype=int)[br]
    GN = BN @ G

    x = np.array([np.dot(u, GN[:, j]) % 2 for j in range(N)], dtype=int)
    return x
