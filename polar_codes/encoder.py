"""
极化码编码器
编码：蝶形结构实现 O(N log N)，与 SC 译码器（按比特倒序相位）配套
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 rev[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无额外输出置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                i = base + k
                u[i] ^= u[i + half]
        block = half
    return u
