"""
极化码编码器
编码：x = u * G_N，G_N = F^{\otimes n}（无额外比特倒序）
"""
import numpy as np


def _generator_matrix(n):
    f2 = np.array([[1, 0], [1, 1]], dtype=int)
    g = f2.copy()
    for _ in range(1, n):
        g = np.kron(g, f2)
    return g


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：x = u @ G_N (mod 2)，G_N = F^{\otimes n}。
    """
    u = np.asarray(u, dtype=int).ravel()
    N = len(u)
    n = int(np.log2(N))
    g = _generator_matrix(n)
    return (u @ g) % 2
