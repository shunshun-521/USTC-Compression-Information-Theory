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
    极化码编码（蝶形递归结构，不做输出比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    step = 1
    while step < len(u):
        for i in range(0, len(u), 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1
    return u


def polar_encode_with_reversal(u):
    """带比特倒序置换的编码（供需要 B_N 置换的场景使用）"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]
