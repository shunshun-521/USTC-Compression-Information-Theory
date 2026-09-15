"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：x = B_N * F^{⊗n} * u
    """
    u = np.array(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    v = u.copy()
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            v[i:i + step] ^= v[i + step:i + 2 * step]
        step *= 2

    br = bit_reversal_permutation(N)
    x = v[br]
    return x.astype(int)
