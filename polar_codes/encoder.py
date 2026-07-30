"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    for stage in range(n):
        block = 1 << stage
        step = block << 1
        for start in range(0, N, step):
            for j in range(block):
                u[start + j] ^= u[start + j + block]

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
