"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码，蝶形结构 O(N log N)。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：每层对左半部分 u[l] ^= u[l+span]（左支 XOR），与 SC 译码器配对。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    span = N
    while span > 1:
        half = span // 2
        for base in range(0, N, span):
            for k in range(half):
                l = base + k
                u[l] ^= u[l + half]
        span = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
