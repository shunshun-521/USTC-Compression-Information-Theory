"""
极化码编码器
编码：x = u * F^{\\otimes n}（Arikan 标准蝶形结构）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（标准非递归 Arikan 蝶形）。

    每层对块内左半部分累加右半部分：u[l] ^= u[l + block/2]
    """
    u = np.asarray(u, dtype=int).copy()
    n = len(u)
    block = n
    while block > 1:
        half = block // 2
        for p in range(0, n, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u


def build_generator_matrix(N):
    """构建 F^{\\otimes n}，用于验证"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(F, G)
    return G
