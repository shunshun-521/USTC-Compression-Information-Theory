"""
极化码编码器
编码：x = u F^{\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(np.int64)
    rev = rev[:, ::-1]
    powers = 1 << np.arange(n)
    return (rev * powers).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（非系统形，x = u F^{\otimes n}）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u.astype(int)


def polar_decode(u_or_x):
    """译码端恢复源向量：F 为自逆，u = x F^{\otimes n}"""
    return polar_encode(u_or_x)
