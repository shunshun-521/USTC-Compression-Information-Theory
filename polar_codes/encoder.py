"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：x = u * G_N，G_N = B_N * F^{\\otimes n}（先蝶形再比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    for s in range(n):
        step = 1 << (s + 1)
        half = step >> 1
        for i in range(0, N, step):
            u[i : i + half] ^= u[i + half : i + step]

    br = bit_reversal_permutation(N)
    return u[br]


def polar_encode_systematic(u):
    """仅蝶形（等价于 x = u * F^{\\otimes n}），供内部一致性测试"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << (s + 1)
        half = step >> 1
        for i in range(0, N, step):
            u[i : i + half] ^= u[i + half : i + step]
    return u
