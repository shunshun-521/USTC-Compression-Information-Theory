"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """对 n 位索引 i 做比特倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def polar_encode(u):
    """
    极化码编码（自顶向下蝶形 XOR，与 PSC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)

    def encode_rec(i1, i2):
        h_shift = (i2 - i1 + 1) // 2
        mid = i1 + h_shift
        for k in range(i1, mid):
            x[k] ^= x[k + h_shift]
        if h_shift >= 2:
            encode_rec(i1, mid - 1)
            encode_rec(mid, i2)

    encode_rec(0, N - 1)
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
