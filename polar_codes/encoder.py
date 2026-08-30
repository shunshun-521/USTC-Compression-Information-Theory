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
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 Permuted SCD 译码器配套）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    m = N
    while m > 1:
        half = m // 2
        for p in range(0, N, m):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        m = half
    return u


def polar_encode_with_bitrev(u):
    """含比特倒序置换的编码（G = B_N F^{\\otimes n}）。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(n):
        step = 2 ** (stage + 1)
        half = step // 2
        for i in range(0, N, step):
            for j in range(i, i + half):
                u[j] ^= u[j + half]

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed:", x)
