"""
极化码编码器
编码：u 经 Kronecker 结构 XOR（与标准 polar_encode 一致，不含额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)
    return rev


def bit_reversed(x, n):
    """单整数比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Kronecker XOR，O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    size = N
    while size > 1:
        n_split = size // 2
        for p in range(0, N, size):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        size = n_split
    return u.astype(int)


def polar_generator_matrix(N):
    """G_N = F^{\\otimes n}（无 B_N）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G % 2
