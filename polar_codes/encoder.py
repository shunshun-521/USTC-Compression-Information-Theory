"""
极化码编码器（与 polarcodes 非递归编码一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r = (r << 1) | ((i >> b) & 1)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：u -> x = u * F^{\otimes n}（模 2，无额外比特倒序）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        n_split = block // 2
        for p in range(0, N, block):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        block = n_split
    return u
