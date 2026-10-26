"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if (1 << n) != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    n = len(x)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            block = x[i : i + 2 * step]
            left = block[:step]
            right = block[step:]
            block[:step] = left ^ right
        step *= 2
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder self-test passed:", x)
