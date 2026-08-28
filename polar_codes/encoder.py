"""
极化码编码器
编码：Arikan 蝶形结构，自然序输出（无比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，自然序输出）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    u2 = np.array([0, 0, 0, 1])
    x2 = polar_encode(u2)
    print("u2 ->", x2)
