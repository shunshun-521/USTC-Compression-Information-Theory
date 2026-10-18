"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    u: 长度为 N 的源序列 -> x: 码字（x = u @ G_N, G_N = B_N F^{⊗n}）
    """
    x = np.array(u, dtype=np.int8).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step <<= 1
    perm = bit_reversal_permutation(N)
    return x[perm]


def build_generator_matrix(N):
    """生成矩阵 G_N = B_N F^{⊗n}（用于校验）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=np.int8)
    perm = bit_reversal_permutation(N)
    for i in range(N):
        B[i, perm[i]] = 1
    return (B @ G) % 2
