"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u * G_N（GF(2)），蝶形 O(N log N) 实现。
    等价于逐层将后半累加到前半：v[0:half] ^= v[half:2*half]。
    """
    u = np.asarray(u, dtype=np.int8).flatten().copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")

    v = u
    for ll in range(1, n + 1):
        n_iter = 1 << ll
        half = n_iter // 2
        for kk in range(N // n_iter):
            a = kk * n_iter
            v[a : a + half] ^= v[a + half : a + n_iter]
    return v.astype(int)
