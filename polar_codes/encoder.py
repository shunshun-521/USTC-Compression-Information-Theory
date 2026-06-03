"""
极化码编码器
编码：x = u * G_N，蝶形左支 XOR（与标准 polarcodes 实现一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def bit_reversed_index(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def polar_encode(u):
    """
    极化码编码（O(N log N) 蝶形，左支 XOR）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n_split = N
    while n_split > 1:
        n_split //= 2
        for p in range(0, N, 2 * n_split):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u=", u, "-> x=", polar_encode(u))
