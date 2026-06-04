"""
极化码编码器
编码：u 经 Arikan 极化变换（蝶形 XOR），与 SC 译码器配套
"""
import numpy as np


def _bit_reverse_index(i, n):
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reverse_index(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half

    return u
