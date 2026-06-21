"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import math

import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(math.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 F^{\\otimes n} 等价，不做额外比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    seq_len = 1
    tree_depth = int(math.log2(N))
    for _ in range(tree_depth):
        for i in range(0, N, 2 * seq_len):
            first = u[i:i + seq_len]
            second = u[i + seq_len:i + 2 * seq_len]
            u[i:i + 2 * seq_len] = np.concatenate([(first + second) % 2, second])
        seq_len *= 2
    return u


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N = F^{\\otimes n}（自然序）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(math.log2(N))
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2
