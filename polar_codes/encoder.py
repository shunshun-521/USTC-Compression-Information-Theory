"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：u[l] ^= u[l + n_split]，与 Arikan 核 F=[[1,1],[0,1]] 一致。
    最后对比特倒序置换后的码字输出。
    """
    x = np.array(u, dtype=int).copy()
    N = len(x)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                x[base + k] ^= x[base + k + half]
        block = half

    br = bit_reversal_permutation(N)
    return x[br]


def polar_encode_core(u):
    """蝶形编码，不做比特倒序（译码器内部使用）"""
    x = np.array(u, dtype=int).copy()
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                x[base + k] ^= x[base + k + half]
        block = half
    return x
