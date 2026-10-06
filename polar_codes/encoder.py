"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（递归蝶形 XOR，与 SC 译码器配套，不在输出端做比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    def encode_partition(i1, i2):
        h_shift = (i2 - i1 + 1) // 2
        mid = i1 + h_shift
        for k in range(i1, mid):
            u[k] ^= u[k + h_shift]
        if h_shift >= 2:
            encode_partition(i1, mid - 1)
            encode_partition(mid, i2)

    encode_partition(0, N - 1)
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"encoder failed: {x}"
