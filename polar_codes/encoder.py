"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N)，输出不做比特倒序（与 SC 译码器配套）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无输出比特倒序）。

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
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_with_reversal(u):
    """带输出比特倒序的编码（备用）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]
