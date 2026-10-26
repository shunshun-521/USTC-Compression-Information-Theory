"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in idx:
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    u: 长度为 N 的源序列（信息位 + 冻结位）
    x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u.astype(np.int8)


def build_generator_matrix(N):
    """GF(2) 生成矩阵 G_N = F^{\\otimes n}（与译码器一致的非倒序约定）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    m = int(np.log2(N))
    for _ in range(m - 1):
        G = np.kron(G, F)
    return G % 2
