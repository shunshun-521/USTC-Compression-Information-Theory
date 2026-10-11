"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N)
"""
import numpy as np


def bit_reversed(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（与标准蝶形 XOR 一致，无额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half
    return u.astype(int)


def polar_generator_matrix(N):
    """由编码器行为生成 G，满足 (u @ G) % 2 = polar_encode(u)。"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        ui = np.zeros(N, dtype=int)
        ui[i] = 1
        G[i] = polar_encode(ui)
    return G.T
