"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(bin(i)[2:].zfill(n)[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    for s in range(1, n + 1):
        block_size = 2 ** s
        half = 2 ** (s - 1)
        num_blocks = N // block_size
        for j in range(num_blocks):
            base = j * block_size
            for i in range(half):
                u[base + i] ^= u[base + half + i]

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x
