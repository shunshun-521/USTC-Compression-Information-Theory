"""
极化码编码器
编码：x = u * F_N（蝶形结构，O(N log N)）
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


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 Permuted SC 译码配套）。
    等价于 x = u * F_N，其中 F_N = F^{\\otimes n}。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i: i + step] ^= u[i + step: i + 2 * step]
        step <<= 1
    return u


def build_generator_matrix(N):
    """构建 F_N = F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G
