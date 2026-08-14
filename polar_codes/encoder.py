"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(np.binary_repr(i, width=n)[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC 译码器配套的索引约定）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for ll in range(1, n + 1):
        n_iter = 1 << ll
        n_half = n_iter // 2
        for kk in range(N // n_iter):
            base = kk * n_iter
            u[base:base + n_half] ^= u[base + n_half:base + n_iter]

    return u
