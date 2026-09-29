"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_int(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversal_int(i, n) for i in range(N)], dtype=np.int64)


def polar_generator_matrix(N):
    """返回 GF(2) 上的极化生成矩阵 G_N（行向量编码 x = u @ G_N）"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        u = np.zeros(N, dtype=np.int8)
        u[i] = 1
        G[i] = polar_encode(u)
    return G


def polar_encode(u):
    """
    极化码编码：蝶形结构（与置换 SC 译码配套，不在输出端做 B_N）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
