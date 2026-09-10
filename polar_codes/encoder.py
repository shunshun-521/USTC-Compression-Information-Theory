"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
    等价于 Arikan 核 F=[[1,1],[0,1]] 的 Kronecker 积，最后做比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] = (u[i + j] ^ u[i + j + step]) & 1
        step *= 2

    br = bit_reversal_permutation(N)
    return u[br]
