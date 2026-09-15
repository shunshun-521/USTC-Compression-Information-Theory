"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：u[l] ^= u[l + half]（分块递归），最后做 bit-reversal。
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                x[p + k] ^= x[p + k + half]
        block = half

    return x[bit_reversal_permutation(N)]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
