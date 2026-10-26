"""
极化码编码器
编码：x = u G_N，利用蝶形结构实现 O(N log N) 复杂度
（与 G_N = F^{⊗n} 的矩阵乘法 u @ G 一致，不含额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u @ G_N（GF(2)），蝶形实现。
    """
    x = np.array(u, dtype=int).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    return x
