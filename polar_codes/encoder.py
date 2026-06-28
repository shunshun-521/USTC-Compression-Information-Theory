"""
极化码编码器
编码：x = u * G_N（Arikan 核 F），蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)],
        dtype=int,
    )


def bit_reversed_index(i, n):
    """单索引比特倒序。"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，与译码器一致的 Arikan 核 F=[[1,1],[0,1]]）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    for _ in range(int(np.log2(N))):
        if block == 1:
            break
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
