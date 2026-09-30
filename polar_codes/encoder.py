"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("N must be power of 2")
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in idx:
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u * G_N, G_N = B_N F^{\\otimes n}
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return x[br]


def polar_encode_no_br(u):
    """仅蝶形，不做比特倒序（用于 BP 早停重编码一致性）"""
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    return x
