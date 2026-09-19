"""
极化码编码器
编码：蝶形 XOR 结构 O(N log N)，无输出比特倒序
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        val = i
        for _ in range(n):
            r = (r << 1) | (val & 1)
            val >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，大粒度到小粒度，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                u[start + k] ^= u[start + k + half]
        block //= 2
    return u
