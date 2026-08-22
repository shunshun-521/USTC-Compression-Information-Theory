"""
极化码编码器：蝶形 XOR（与 SCD 译码器配套）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for j in range(n):
            if (i >> j) & 1:
                r |= 1 << (n - 1 - j)
        rev[i] = r
    return rev


def polar_encode(u):
    """极化码编码（蝶形 XOR）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i] ^= u[i + step]
        step *= 2
    return u
