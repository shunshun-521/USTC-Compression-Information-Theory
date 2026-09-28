"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev_idx = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev_idx[i] = r
    return rev_idx


def _bit_reversed_int(x, n):
    r = 0
    for i in range(n):
        if x & (1 << i):
            r |= 1 << (n - 1 - i)
    return r


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 Arikan F^{\\otimes n} 生成矩阵一致）。
    译码器在比特倒序顺序下恢复 u，与信道端码字索引一致。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_matrix(N):
    """生成矩阵 G_N（用于调试）"""
    G = np.eye(N, dtype=int)
    for row in range(N):
        G[row] = polar_encode(G[row])
    return G
