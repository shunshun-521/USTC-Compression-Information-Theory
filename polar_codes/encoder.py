"""
极化码编码器
编码：x = u * F^(⊗n)，利用蝶形结构实现 O(N log N) 复杂度
（与置换 SC 译码器配套，等价于 G_N = B_N * F^(⊗n) 约定）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_rev_indices(N)


def _bit_rev_indices(N):
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r = (r << 1) | ((i >> b) & 1)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR 结构，O(N log N)）。

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
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    u2 = np.array([0, 1, 0, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [0, 0, 1, 1]), f"编码器错误: {x2}"
