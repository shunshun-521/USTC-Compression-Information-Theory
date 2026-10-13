"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：u[i] ^= u[i+step]，共 log2(N) 层；输出按比特倒序重排。
    """
    u = np.array(u, dtype=np.uint8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    brp = bit_reversal_permutation(N)
    return u[brp]


def polar_encode_matrix(u):
    """GF(2) 矩阵乘法编码，用于校验"""
    from construction import build_generator_matrix

    N = len(u)
    G = build_generator_matrix(N)
    return (np.array(u, dtype=np.uint8) @ G) % 2
