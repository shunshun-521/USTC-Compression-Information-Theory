"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in indices], dtype=int)


def polar_encode_core(u):
    """蝶形编码（不含比特倒序）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 2 ** (stage + 1)
        half = step // 2
        for i in range(0, N, step):
            for j in range(half):
                u[i + j] ^= u[i + j + half]
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    返回信道传输顺序的码字。
    """
    x = polar_encode_core(u)
    br = bit_reversal_permutation(len(x))
    return x[br]
