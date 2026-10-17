"""
极化码编码器
编码：蝶形 u[i] ^= u[i+step]，最后对比特倒序置换（与 permuted SC 译码配套）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 + 比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("Length of u must be a power of 2")

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def generator_matrix(N):
    """由 polar_encode 列向量构造生成矩阵（mod 2）。"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[:, i] = polar_encode(e)
    return G


def verify_encoder_unit_test():
    """与生成矩阵一致的编码校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    expected = (G @ u) % 2
    assert np.array_equal(x, expected), f"编码器错误: got {x}, expected {expected}"
    u2 = np.array([0, 1, 0, 0])
    assert np.array_equal(polar_encode(u2), (G @ u2) % 2)
