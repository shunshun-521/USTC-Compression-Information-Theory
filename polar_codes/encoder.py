"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode_core(u):
    """
    极化码蝶形编码（不含比特倒序），F=[[1,0],[1,1]] 约定。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    length = len(u)
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            u[start:start + step] ^= u[start + step:start + 2 * step]
        step *= 2
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u * B_N * F^{⊗n}
    """
    x = polar_encode_core(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def polar_encode_no_br(u):
    """不含比特倒序置换的编码"""
    return polar_encode_core(u)


def build_generator_matrix(N):
    """从编码器构造 G_N，满足 polar_encode_no_br(u) = (u @ G_N) % 2"""
    n = int(np.log2(N))
    G = np.zeros((N, N), dtype=int)
    for j in range(N):
        u = np.zeros(N, dtype=int)
        u[j] = 1
        G[j] = polar_encode_no_br(u)
    return G


def build_generator_matrix_with_br(N):
    """含比特倒序的生成矩阵"""
    n = int(np.log2(N))
    G = np.zeros((N, N), dtype=int)
    for j in range(N):
        u = np.zeros(N, dtype=int)
        u[j] = 1
        G[j] = polar_encode(u)
    return G
