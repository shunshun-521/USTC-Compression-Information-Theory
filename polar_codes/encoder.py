"""
极化码编码器
编码：x = u * F^⊗n（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=np.int64)
    for i in range(N):
        v, r = i, 0
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（Kronecker F^⊗n，与 SC/SCL/BP 译码器一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
    return u
