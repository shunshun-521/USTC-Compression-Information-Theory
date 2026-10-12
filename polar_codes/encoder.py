"""
极化码编码器
编码：u 经 F^⊗n 蝶形变换得到码字（与 SC 因子图一致，无额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 rev，满足 out[i] = in[rev[i]]。"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for b in range(n):
        rev |= ((indices >> b) & 1) << (n - 1 - b)
    return rev


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（非系统化，O(N log N) 蝶形）。
    """
    u = np.asarray(u, dtype=int).copy()
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
