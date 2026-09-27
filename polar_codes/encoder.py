"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N)
"""
import numpy as np


def bit_reversal_index(x, n):
    """单索引比特倒序（用于译码相位顺序）"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversal_index(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 因子图一致，输出即为信道发送比特顺序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("Length must be a power of 2")
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def build_generator_matrix(N):
    """构造生成矩阵（用于校验）"""
    G = np.zeros((N, N), dtype=int)
    for j in range(N):
        ej = np.zeros(N, dtype=int)
        ej[j] = 1
        G[j] = polar_encode(ej)
    return G
