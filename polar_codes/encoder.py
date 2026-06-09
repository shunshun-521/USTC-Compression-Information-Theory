"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形左支路 XOR，与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half
    return u.astype(int)


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于校验，与 polar_encode 等价）"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
