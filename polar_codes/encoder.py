"""
极化码编码器
编码：x = u * F^{\\otimes n}，蝶形 XOR，不做输出比特倒序
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for b in range(n):
        rev = (rev << 1) | ((indices >> b) & 1)
    return rev


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for block in range(0, N, n):
            for k in range(half):
                idx = block + k
                u[idx] ^= u[idx + half]
        n = half
    return u


def polar_generator_matrix(N):
    """生成矩阵 F^{\\otimes n}（用于测试）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G
