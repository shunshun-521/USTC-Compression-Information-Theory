"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    x = u.copy()
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            x[i:i + step] ^= x[i + step:i + 2 * step]

    br = bit_reversal_permutation(N)
    return x[br]


def polar_encode_no_br(u):
    """不含比特倒序的编码（内部使用）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    x = u.copy()
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            x[i:i + step] ^= x[i + step:i + 2 * step]
    return x
