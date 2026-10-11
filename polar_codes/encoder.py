"""
极化码编码器
编码：x = u * G_N，蝶形结构 + 比特倒序置换
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_transform(u):
    """Kronecker F^{⊗n} 蝶形变换（无比特倒序）"""
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x


def polar_encode(u):
    """
    极化码编码：蝶形变换后做比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    x = polar_transform(u)
    br = bit_reversal_permutation(N)
    return x[br]
