"""
极化码编码器
编码：x = u * F_N（蝶形结构，O(N log N)）
比特倒序在译码阶段处理，与标准 SCD 实现一致。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed_index(x, n):
    """对标量索引 x 做 n 位比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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
    nblk = N
    for _ in range(int(np.log2(N))):
        if nblk == 1:
            break
        n_split = nblk // 2
        for p in range(0, N, nblk):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        nblk = n_split
    return u
