"""
极化码编码器
编码：u 经蝶形 XOR（无输出比特倒序），与标准 SCD 译码配套
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int32)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。
    按分区大小从 N 递减至 2，将右半分区 XOR 到左半分区。
    """
    u = np.asarray(u, dtype=np.int8)
    x = u.copy()
    N = len(x)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                x[l] ^= x[l + n_split]
        n = n_split
    return x
