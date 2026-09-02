r"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = B_N F^{\otimes n}
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
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = a + step
                u[a] = (u[a] + u[b]) % 2
        step *= 2
    brp = bit_reversal_permutation(N)
    return u[brp]


def build_generator_matrix(N):
    """构建 G_N = B_N F^otimes n，用于验证"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    brp = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(brp):
        B[i, j] = 1
    return (B @ G) % 2


def polar_encode_matrix(u):
    """矩阵乘法编码（验证用）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
