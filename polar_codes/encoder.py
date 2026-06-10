"""
极化码编码器
编码：x = u @ F^⊗n，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ G_N (mod 2)
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    block = N >> 1
    while block > 0:
        for j in range(0, N, 2 * block):
            for i in range(block):
                x[j + i] ^= x[j + i + block]
        block >>= 1
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
