"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（Kronecker F^{\\otimes n} 蝶形结构，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            u[start : start + step] ^= u[start + step : start + 2 * step]
        step <<= 1
    return u


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}，用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F) % 2
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码（校验用）。"""
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
