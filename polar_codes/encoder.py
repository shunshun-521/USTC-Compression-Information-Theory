"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev[:, ::-1].dot(1 << np.arange(n))
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ G_N (mod 2)
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    step = 1
    for _ in range(n):
        for i in range(0, len(u), 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1
    br = bit_reversal_permutation(len(u))
    return u[br]
