"""
极化码编码器：Arikan 蝶形结构，O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构）。
    x = u * F^{\otimes n}
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, len(u), 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u
