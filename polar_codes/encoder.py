"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int_)
    rev = np.zeros(N, dtype=np.int_)
    for bit in range(n):
        rev = (rev << 1) | ((indices >> bit) & 1)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int_)
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    v = u.copy()
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            v[i:i + step] = v[i:i + step] ^ v[i + step:i + 2 * step]
        step *= 2

    brp = bit_reversal_permutation(N)
    return v[brp]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([0, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [0, 0, 1, 1]), f"编码器错误: {x2}"
    print("编码器校验通过")
