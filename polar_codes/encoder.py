"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（GF(2) 下 x = u * G_N）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("u length must be a power of 2")
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << s
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u.astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}"""
    from _codec_core import generate_matrix

    n = int(np.log2(N))
    return generate_matrix(n)
