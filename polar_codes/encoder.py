"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev += ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，不含比特倒序置换）。
    与 SC 译码器配套：译码按比特倒序处理各比特位。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2
    return u


def polar_encode_with_br(u):
    """带比特倒序置换的编码（x = u * G_N, G_N = B_N F^{⊗n}）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]
