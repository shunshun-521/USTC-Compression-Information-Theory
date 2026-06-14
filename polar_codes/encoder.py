"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}，F = [[1,0],[1,1]]
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（O(N log N) 蝶形结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    if len(u) != 2 ** n:
        raise ValueError("Length of u must be a power of 2")

    step = 1
    while step < len(u):
        for i in range(0, len(u), 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    return u


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（GF(2)），用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2
