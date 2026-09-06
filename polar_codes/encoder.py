"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N) 复杂度（Arikan 核 [[1,1],[0,1]]）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无输出比特倒序）。

    与标准 Arikan 核 F=[[1,1],[0,1]] 的 u @ F^{\otimes n} 等价。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    stage_n = N
    while stage_n > 1:
        n_split = stage_n // 2
        for p in range(0, N, stage_n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        stage_n = n_split
    return u


def polar_generator_matrix(N):
    """返回 N x N 生成矩阵 F^{\otimes n}（Arikan 核）"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    F_n = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        F_n = np.kron(F, F_n)
    return F_n
