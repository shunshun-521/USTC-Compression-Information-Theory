"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx >> np.arange(n)) & 1).sum(axis=1) if False else None
    # vectorized bit reversal
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _bit_rev_indices(N):
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n)) & 1
    rev_bits = bits[:, ::-1]
    powers = 1 << np.arange(n)
    return (rev_bits * powers).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    step = 1
    n = int(np.log2(N))
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    br = _bit_rev_indices(N)
    return u[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
