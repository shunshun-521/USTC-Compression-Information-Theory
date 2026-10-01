"""
极化码编码器
编码：u 经蝶形 XOR（与 F^{⊗n} 等价），不在输出端做比特倒序。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int64)
    rev = rev[:, ::-1]
    powers = 2 ** np.arange(n)
    return (rev * powers).sum(axis=1)


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = N
    while step > 1:
        step //= 2
        for p in range(0, N, step * 2):
            u[p : p + step] ^= u[p + step : p + 2 * step]
    return u.astype(int)
