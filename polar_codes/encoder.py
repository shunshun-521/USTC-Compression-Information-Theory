"""
极化码编码器
编码：利用蝶形结构实现 O(N log N) 复杂度
与 SC/SCL/BP 译码器配套（译码按比特倒序处理）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，u[i] ^= u[i+step]）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n_len = len(u)
    if n_len & (n_len - 1):
        raise ValueError("Length must be a power of 2")

    block = n_len
    while block > 1:
        half = block // 2
        for base in range(0, n_len, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half

    return u.astype(int)
