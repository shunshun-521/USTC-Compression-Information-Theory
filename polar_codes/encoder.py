"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序（与 polarcodes.utils.bit_reversed 一致）。"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode_butterfly(u):
    """蝶形编码（无输出比特倒序），x = u * F^{\\otimes n}。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码：蝶形变换后做比特倒序置换，码字满足 x = u * G_N，G_N = B_N F^{\\otimes n}。
    """
    x = polar_encode_butterfly(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def polar_generator_matrix(N):
    """返回 G_N = B_N F^{\\otimes n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i, p in enumerate(br):
        B[i, p] = 1
    return (B @ G) % 2
