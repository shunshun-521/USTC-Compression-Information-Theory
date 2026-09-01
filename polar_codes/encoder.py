"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{⊗n}（与 3GPP / Arikan 标准 Kronecker 构造一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（蝶形结构，不含额外比特倒序）。
    x = u * G_N，G_N = F^{⊗n}
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] = u[i:i + step] ^ u[i + step:i + 2 * step]
        step *= 2
    return u


def get_G_N(N):
    """生成极化码生成矩阵 G_N = F^{⊗n}"""
    G = np.array([[1, 0], [1, 1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, np.array([[1, 0], [1, 1]], dtype=int))
    return G
