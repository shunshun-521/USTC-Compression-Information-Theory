"""
极化码编码器
编码：x = u * F_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 生成矩阵 F^⊗n）。

    蝶形结构：u[l] ^= u[l + step]（上层更新），分 log2(N) 个阶段。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        block = half
    return u
