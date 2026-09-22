"""
极化码编码器
蝶形结构 O(N log N)，与 Permuted SC 译码器配套
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码蝶形编码（无输出比特倒序）。
    与 Permuted SC 译码器配套使用。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_standard_br(u):
    """标准 G_N 编码（蝶形 + 比特倒序），备用"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


def polar_encode_with_br(u):
    return polar_encode_standard_br(u)
