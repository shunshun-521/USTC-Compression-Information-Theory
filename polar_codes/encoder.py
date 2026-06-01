"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 perm，满足 out[i] = in[perm[i]]"""
    n = int(np.log2(N))
    perm = np.arange(N, dtype=int)
    for i in range(N):
        rev = 0
        v = i
        for _ in range(n):
            rev = (rev << 1) | (v & 1)
            v >>= 1
        perm[i] = rev
    return perm


def polar_encode(u):
    """
    极化码编码：蝶形 u[i] ^= u[i+step]，共 log2(N) 层。
    比特倒序由 GA 构造中的可靠性排序与 SC 译码索引约定统一处理。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
