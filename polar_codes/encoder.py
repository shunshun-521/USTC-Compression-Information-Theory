"""
极化码编码器
编码：x = u * F^⊗n，利用蝶形结构实现 O(N log N) 复杂度
（配合 Permuted SC 译码器，不在编码端做比特倒序置换）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(np.binary_repr(i, width=n)[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，无末尾比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = a + step
                u[a] ^= u[b]
        step <<= 1
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
