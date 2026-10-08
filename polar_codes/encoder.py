"""
极化码编码器
编码：x = u * F^⊗n（自然序，与 SC/SCL 译码器一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
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
    极化码 Kronecker 编码（蝶形 XOR，无比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            u[start : start + step] ^= u[start + step : start + 2 * step]
        step <<= 1
    return (u % 2).astype(int)


def polar_encode_matrix(u):
    """矩阵法编码（校验用）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
