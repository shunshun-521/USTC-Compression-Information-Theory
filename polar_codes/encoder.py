"""
极化码编码器
编码：x = u * G_N（F^{\otimes n}），蝶形 XOR 结构，O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def arikan_generator(N):
    """生成矩阵 F^{\otimes n}（GF(2)）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    return G.astype(np.int8)


def polar_encode(u):
    """
    极化码编码（与译码器一致的非比特倒序 Arikan 约定）。
    蝶形：x[start:start+step] ^= x[start+step:start+2*step]
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = x.size
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start : start + step] = (
                x[start : start + step] + x[start + step : start + 2 * step]
            ) % 2
        step <<= 1
    return x.astype(int)


def polar_encode_with_bit_reversal(u):
    """u @ G_N 后再做比特倒序置换（教材形式，与默认译码需配合 LLR 置换）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


def polar_encode_matrix(u):
    """矩阵法编码（校验）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    return (u @ arikan_generator(N)) % 2
