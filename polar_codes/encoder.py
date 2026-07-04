"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构后输出经比特倒序置换，与 G_N = B_N F^{\\otimes n} 一致。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N

    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            u[start : start + half] ^= u[start + half : start + block]
        block = half

    br = bit_reversal_permutation(N)
    return u[br]


def polar_encode_butterfly(u):
    """仅蝶形编码（不做比特倒序），供内部一致性测试"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            u[start : start + half] ^= u[start + half : start + block]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
