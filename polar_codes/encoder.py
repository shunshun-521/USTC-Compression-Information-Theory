"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
使用 Arikan 核 F = [[1,1],[0,1]]，无输出比特倒序
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码。

    蝶形结构：每阶段 u[l] ^= u[l + n_split]（左分支累加右分支），
    阶段长度从 N 减半至 1。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] = u[l] ^ u[l + n_split]
        n = n_split
    return u


def build_generator_matrix(N):
    """构建 G_N = F^{⊗n}，F = [[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    F_n = np.array([[1]], dtype=int)
    n = int(np.log2(N))
    for _ in range(n):
        F_n = np.kron(F_n, F)
    return F_n
