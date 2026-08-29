"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def _butterfly_encode(u):
    """蝶形变换（不含比特倒序）。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = N
    while step > 1:
        half = step // 2
        for p in range(0, N, step):
            u[p : p + half] ^= u[p + half : p + step]
        step = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int)
    rev = bit_reversal_permutation(len(u))
    return _butterfly_encode(u)[rev]


def polar_encode_matrix(u):
    """基于生成矩阵 G_N = B_N F^{\\otimes n} 的编码（用于验证）。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    G_N = (np.eye(N, dtype=int)[rev] @ G) % 2
    return (u @ G_N) % 2
