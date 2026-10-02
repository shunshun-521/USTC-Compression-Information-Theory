"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    u: 长度为 N 的源序列（信息位 + 冻结位）
    返回码字 x，长度 N
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << s
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    br = bit_reversal_permutation(N)
    return x[br].astype(int)


def build_generator_matrix(N):
    """生成矩阵 G，第 i 行为 polar_encode(e_i)"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[i] = 1
        G[i] = polar_encode(e)
    return G
