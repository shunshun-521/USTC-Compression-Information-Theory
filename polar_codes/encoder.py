"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    等价于 x = u * B_N * F^{\\otimes n} (mod 2)。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, step << 1):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_encode_no_br(u):
    """蝶形编码，不做输出比特倒序（与内部译码树索引一致）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, step << 1):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u.astype(int)
