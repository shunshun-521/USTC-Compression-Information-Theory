"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int
    )


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u * B_N * F^{\\otimes n}
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2

    br = bit_reversal_permutation(N)
    return u[br]
