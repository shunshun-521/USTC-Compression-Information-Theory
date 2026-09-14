"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n} B_N
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR + 比特倒序置换）。

    每层对块内所有对 (i+j, i+j+step) 执行 u[i+j] ^= u[i+j+step]，
    最后对比特倒序索引重排输出。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 2 ** stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u[bit_reversal_permutation(N)]


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n} B_N"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[:, br]
    return (G @ B) % 2
