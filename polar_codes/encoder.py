"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗ n}；蝶形实现 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])


def _bit_reversed_int(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（蝶形结构，与 SC 译码器默认配套）。

    参数：
        u: 长度为 N 的源序列
        apply_bit_reversal: 若为 True，对输出做比特倒序（G_N = B_N F^{⊗n}）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 2**layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    if apply_bit_reversal:
        rev = bit_reversal_permutation(N)
        return u[rev]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
