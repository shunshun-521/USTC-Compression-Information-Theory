"""
极化码编码器
编码：x = u * G_N，G_N = B_N * F^{\otimes n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[i] = u[i] XOR u[i + step]
        - 共 log2(N) 层
        - 最后做比特倒序置换
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(n):
        step = 2 ** stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                u[a] ^= u[b]

    br = bit_reversal_permutation(N)
    x = u[br]
    return x


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\otimes n}，用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    nn = int(np.log2(N))
    for _ in range(nn - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    return B @ G % 2
