"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    采用标准 Kronecker 蝶形结构：u[l] ^= u[l + block_size]，
    block_size 从 N/2 递减至 1（与 Arikan 生成矩阵一致）。
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("Length of u must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                x[idx] ^= x[idx + half]
        block = half
    return x
