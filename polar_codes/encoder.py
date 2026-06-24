"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """将 i 的 n 位二进制表示倒序后转回整数。"""
    return int(f"{i:0{n}b}"[::-1], 2)


def polar_encode(u):
    """
    极化码编码（自然顺序蝶形，与 mcba1n 参考实现一致）。
    每层：u[l] ^= u[l + n_split]
    """
    u = np.asarray(u, dtype=np.int_).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u
