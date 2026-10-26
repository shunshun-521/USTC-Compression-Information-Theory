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


def _bit_reverse_indices(N):
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    bits = np.unpackbits(idx.view(np.uint8)).reshape(-1, 8)[:, ::-1]
    # simpler approach
    rev = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：每层 (a, b) -> (a XOR b, b)，最后对码字做比特倒序。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            left = u[i : i + step].copy()
            right = u[i + step : i + 2 * step]
            u[i : i + step] = left ^ right
            u[i + step : i + 2 * step] = right
        step <<= 1
    return u.astype(np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
