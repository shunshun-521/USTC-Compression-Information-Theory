"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序。"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构计算 u * F^{⊗n}，最后对比特倒序置换以匹配标准生成矩阵 G_N = B_N F^{⊗n}。
    """
    x = np.array(u, dtype=int, copy=True)
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                x[idx] ^= x[idx + half]
        block = half

    return x[bit_reversal_permutation(N)]


def butterfly_encode(u):
    """仅蝶形编码（无比特倒序），供译码器内部校验使用。"""
    x = np.array(u, dtype=int, copy=True)
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                x[idx] ^= x[idx + half]
        block = half
    return x
