"""
极化码编码器
编码：u * F^⊗n（蝶形结构，O(N log N)）
与 Vangala 置换 SC 译码器配套（不在此处做比特倒序置换）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码蝶形编码（与 polarcodes / Vangala SC 一致的索引约定）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                x[p + k] ^= x[p + k + n_split]
        n = n_split
    return x.astype(np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u=", u, "-> x=", polar_encode(u))
