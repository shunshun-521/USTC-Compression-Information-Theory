"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形： (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    等价于 x[br(i)] = (F^{\\otimes n} u)_i
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = u.shape[0]
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = u[i + j]
                b = u[i + j + step]
                u[i + j] = (a + b) & 1
                u[i + j + step] = b
    # 与 Permuted SCD / G@u 一致（不在此处做比特倒序；倒序在译码树中处理）
    return u


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}（与 polar_encode 一致）"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2
