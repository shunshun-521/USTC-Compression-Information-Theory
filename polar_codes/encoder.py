"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reverse_index(i, n):
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reverse_index(i, n) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形编码 u * F^⊗n（无比特倒序）"""
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for j in range(half):
                u[base + j] ^= u[base + j + half]
        n = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u * B_N * F^⊗n
    """
    x = _butterfly_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def generator_matrix(N):
    """返回 G_N = B_N * F^⊗n（GF(2)）"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    Fn = F
    for _ in range(n - 1):
        Fn = np.kron(F, Fn)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)
    B = B[br]
    return (B @ Fn) % 2
