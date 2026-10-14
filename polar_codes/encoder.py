"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def inverse_bit_reversal_permutation(N):
    """逆比特倒序置换索引"""
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=np.int64)
    inv[br] = np.arange(N)
    return inv


def polar_encode(u):
    """
    极化码编码（蝶形 + 比特倒序置换）。
    x = (u @ F^{\\otimes n}) 的蝶形实现，输出经比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("N=4 encode:", x)
    assert np.array_equal(x, [1, 0, 1, 1])
