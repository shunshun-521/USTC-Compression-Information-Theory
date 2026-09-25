"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    for step in [2 ** i for i in range(n)]:
        for i in range(0, len(u), 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    br = bit_reversal_permutation(len(u))
    x = np.zeros_like(u)
    for i in range(len(u)):
        x[br[i]] = u[i]
    return x


def polar_encode_core(u):
    """蝶形编码（不含比特倒序），供 BP 早停重编码使用。"""
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    for step in [2 ** i for i in range(n)]:
        for i in range(0, len(u), 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u
