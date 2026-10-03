"""
极化码编码器
编码：x = u * F_N（蝶形 XOR，与 SC 译码器配套）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    bits = (idx[:, None] >> np.arange(n)) & 1
    weights = 2 ** np.arange(n - 1, -1, -1)
    return bits.dot(weights).astype(int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    与 decoder_sc 使用相同的 Arikan 极化顺序（不在末尾额外做比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
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
