"""
极化码编码器
编码：x = u * F^{\otimes n}，蝶形结构 O(N log N)
（与 SC/SCL/BP 译码器因子图一致，不含 B_N 置换）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：蝶形 XOR，等价于 u @ F^{\otimes n}（模 2）。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = u.size
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    return u


def polar_generator_matrix(N):
    """生成矩阵 F^{\otimes n}（模 2）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G
