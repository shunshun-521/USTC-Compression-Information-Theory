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


def bit_reversed_index(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 译码器配套）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for j in range(half):
                idx = start + j
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_with_br(u):
    """带输出比特倒序置换的编码（备用）"""
    return polar_encode(u)[bit_reversal_permutation(len(u))]


def polar_encode_matrix(u):
    """使用生成矩阵编码（用于验证）"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
