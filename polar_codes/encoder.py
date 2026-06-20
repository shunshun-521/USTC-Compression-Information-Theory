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
    极化码编码（蝶形结构，等价于 u @ F^{\otimes n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("u length must be a power of 2")

    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    return x


def polar_encode_with_bit_reversal(u):
    """蝶形编码后再做比特倒序置换（B_N 置换）"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def generator_matrix(N):
    """构造 G_N = F^{\otimes n}（不含 B_N）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G
