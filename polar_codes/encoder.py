"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    采用块蝶形结构：每层将下半块 XOR 到上半块（v[upper] ^= v[lower]）。
    与 SC/SCL 译码器配套，不做额外比特倒序。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    for layer in range(1, n + 1):
        block = 1 << layer
        half = block // 2
        for blk in range(N // block):
            start = blk * block
            u[start:start + half] ^= u[start + half:start + block]
    return u
