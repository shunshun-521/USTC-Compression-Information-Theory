"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，不含比特倒序置换）。

    与 SC 译码器（比特倒序相位顺序）配套使用。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N

    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for j in range(start, start + half):
                u[j] ^= u[j + half]
        block = half

    return u


def polar_encode_with_br(u):
    """编码并在输出端施加比特倒序置换。"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]
