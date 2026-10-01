"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 idx，满足 out[i] = in[idx[i]]"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码：x = u * F_N（GF(2)），蝶形 XOR 实现。
    与 decoder / BP 早停中的 polar_encode 保持一致（自然比特序）。
    """
    v = np.array(u, dtype=np.int8, copy=True)
    N = len(v)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                v[i + j] ^= v[i + j + step]
        step <<= 1
    return v.astype(int)


def polar_encode_channel(u):
    """经比特倒序置换后的信道发送序列（部分文献约定）"""
    v = polar_encode(u)
    br = bit_reversal_permutation(len(v))
    return v[br]


def polar_encode_matrix(u):
    """矩阵形式编码（用于校验），u 为行向量"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    G = G.astype(np.int8) & 1
    return (np.array(u, dtype=np.int8) @ G) & 1
