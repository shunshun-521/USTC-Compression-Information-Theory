"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
（与 Permuted SCD 一致：蝶形后不做额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx >> np.arange(n)) & 1).sum(axis=0) if False else None
    rev = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与生成矩阵 B_N F^{⊗n} 一致）。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, step << 1):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u.astype(int)


def polar_generator_matrix(N):
    """G_N = F^{⊗n}（蝶形编码 x = u @ G_N mod 2）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        Fn = np.kron(Fn, F)
    return Fn


def encode_via_matrix(u):
    G = polar_generator_matrix(len(u))
    return (np.array(u, dtype=np.int8) @ G) % 2
