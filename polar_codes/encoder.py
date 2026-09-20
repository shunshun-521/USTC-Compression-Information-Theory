"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N（递归构造）"""
    n = int(np.log2(N))
    if n == 1:
        return np.array([[1, 0], [1, 1]], dtype=np.int8)
    G_half = build_generator_matrix(N // 2)
    k = N // 2
    top = np.hstack([G_half, np.zeros((k, k), dtype=np.int8)])
    bottom = np.hstack([G_half, G_half])
    return np.vstack([top, bottom])


def polar_encode(u):
    """
    极化码编码：蝶形 XOR（与 Permuted SCD 译码器配套）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N

    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_matrix(u):
    """矩阵编码（用于验证）：c = (u @ G) 的比特倒序形式"""
    u = np.asarray(u, dtype=np.int8)
    G = build_generator_matrix(len(u))
    c = (G @ u) % 2
    return c[bit_reversal_permutation(len(u))]
