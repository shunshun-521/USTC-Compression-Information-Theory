"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_value(x, n):
    """单整数比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.array(u, dtype=int).copy()
    n_len = len(u)
    step = 1
    while step < n_len:
        for i in range(0, n_len, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1

    br = bit_reversal_permutation(n_len)
    return u[br]


def polar_encode_matrix(u):
    """基于生成矩阵 B_N F^{\\otimes n} 的编码（用于验证）。"""
    u = np.array(u, dtype=int)
    n = len(u)
    levels = int(np.log2(n))
    f = np.array([[1, 0], [1, 1]], dtype=int)
    g = f.copy()
    for _ in range(1, levels):
        g = np.kron(g, f)
    br = bit_reversal_permutation(n)
    g = g[br, :]
    return (u @ g) & 1
