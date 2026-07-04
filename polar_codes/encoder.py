"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
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


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形：u[i] ^= u[i+step]（GF(2)），共 log2(N) 层，最后比特倒序。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1

    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_encode_nobr(u):
    """无比特倒序编码（内部调试用）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    nn = N
    for _ in range(int(np.log2(N))):
        n_split = nn // 2
        for p in range(0, N, nn):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        nn = n_split
    return u
