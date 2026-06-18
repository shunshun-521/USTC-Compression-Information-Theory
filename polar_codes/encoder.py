"""
极化码编码器
编码：x = u * F_N^{⊗n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与信道自然顺序一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
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


def polar_encode_matrix(u):
    """基于生成矩阵 F^{⊗n} 的编码，用于校验。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(F, G)
    return (u @ G) % 2
