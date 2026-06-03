"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def bit_reversal_index(x, n):
    """单整数比特倒序（0 <= x < 2^n）。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码：左支 XOR 蝶形 + 比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    if len(u) != 2 ** n:
        raise ValueError("u length must be a power of 2")

    block = len(u)
    while block > 1:
        half = block // 2
        for p in range(0, len(u), block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
