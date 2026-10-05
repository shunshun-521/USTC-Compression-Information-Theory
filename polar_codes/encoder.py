"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    assert 2**n == N
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if (i >> bit) & 1:
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，O(N log N)）。
    与 SC 译码配套：译码按比特倒序处理，等价于 x = u B_N F^{⊗n}。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    for _ in range(N):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u.astype(int)
