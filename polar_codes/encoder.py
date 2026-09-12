"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（非系统化，与 mcba1n/polar-codes 一致）。
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    block = len(u)
    for _ in range(n):
        half = block // 2
        for p in range(0, len(u), block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u
