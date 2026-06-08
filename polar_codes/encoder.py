"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 译码器配套）。

    实现与 F^{\\otimes n} 矩阵乘法等价；比特倒序在译码阶段通过
    译码顺序处理（等效于 x = B_N * F^{\\otimes n} * u）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    for _ in range(n):
        half = block // 2
        for p in range(0, len(u), block):
            for k in range(half):
                u[p + k] = u[p + k] ^ u[p + k + half]
        block = half

    return u


def polar_encode_with_br(u):
    """显式比特倒序置换后的码字（用于对照验证）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]
