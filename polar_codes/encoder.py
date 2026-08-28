"""
极化码编码器
编码：x = u * F_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，无输出比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    block = len(u)
    for _ in range(n):
        half = block // 2
        for start in range(0, len(u), block):
            left = slice(start, start + half)
            right = slice(start + half, start + block)
            u[left] ^= u[right]
        block = half
    return u


def build_generator_matrix(N):
    """构建 G_N = F^{\\otimes n}"""
    f = np.array([[1, 0], [1, 1]], dtype=np.int8)
    g = f.copy()
    while g.shape[0] < N:
        g = np.kron(g, f)
    return g
