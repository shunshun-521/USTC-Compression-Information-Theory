"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度

采用与 SC/SCL 译码器一致的 Arikan 核 F=[[1,1],[0,1]] 蝶形结构
（等价于 upper-butterfly：u[i] ^= u[i+step]）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（O(N log N) 蝶形结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        n = half
    return u


def polar_generate_matrix(N):
    """生成 Arikan 生成矩阵 F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G
