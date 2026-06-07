"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形左支 XOR，与 SC 译码器配套）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(1, n + 1):
        block = 1 << layer
        half = block // 2
        for start in range(0, N, block):
            u[start : start + half] ^= u[start + half : start + block]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
