"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    n = int(np.log2(N))
    for s in range(n):
        step = 2 ** s
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
    br = bit_reversal_permutation(N)
    return x[br]


def _self_test():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


if __name__ == "__main__":
    _self_test()
    print("encoder self-test passed")
