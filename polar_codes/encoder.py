"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
（与 Permuted SCD 译码器配套，输出不做比特倒序置换）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed(i, n):
    """对标量索引做比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 Permuted SCD 译码配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1
    return u


def polar_encode_with_reversal(u):
    """带输出比特倒序的编码（G_N = B_N F^{\\otimes n}）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]


def polar_encode_matrix(u):
    """矩阵法编码（F^{\\otimes n}，用于验证）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(F, G)
    return (u @ G) % 2
