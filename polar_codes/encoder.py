r"""
极化码编码器：x = G_N u（G_N = F^{\otimes n}），蝶形 XOR 实现
"""
import numpy as np


def bit_reversal_permutation(N):
    """比特倒序置换索引"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(np.binary_repr(i, width=n)[::-1], 2)
    return rev


def polar_encode(u):
    """极化码编码 O(N log N)"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step <<= 1
    return u
