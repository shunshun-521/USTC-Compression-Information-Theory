"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np
from scd_core import bit_reversal_permutation


def polar_encode_core(u):
    """
    Arikan 蝶形编码（递减分块 XOR，与 SCD 因子图一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n = n_split
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    先蝶形编码，再对码字做比特倒序置换后发送。
    """
    c = polar_encode_core(u)
    br = bit_reversal_permutation(len(c))
    return c[br]


def polar_encode_matrix(u):
    """使用生成矩阵编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        B[i, br[i]] = 1
    G = (B @ F_n) % 2
    return (u @ G) % 2
