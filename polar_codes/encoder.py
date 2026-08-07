"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """单个索引的比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，无输出比特倒序）。
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError(f"Length N={N} must be a power of 2")

    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n = n_split

    return u


def build_generator_matrix(N):
    """构建生成矩阵 F^{⊗n}，F = [[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)

    def kron_power(n):
        if n == 0:
            return np.array([[1]], dtype=int)
        return np.kron(F, kron_power(n - 1))

    return kron_power(int(np.log2(N))) % 2
