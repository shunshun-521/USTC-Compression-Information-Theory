"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """单整数比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（GF(2)）。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        Fn = np.kron(Fn, F)
    br = bit_reversal_permutation(N)
    return Fn[br, :]


def polar_encode(u):
    """
    极化码编码（蝶形 XOR + 比特倒序置换输出）。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    x = u.copy()
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    br = bit_reversal_permutation(N)
    return x[br]


def polar_encode_core(u):
    """蝶形编码（无输出倒序），与置换 SC 因子图一致。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    return x


def polar_encode_matrix(u):
    """矩阵形式编码（校验用）。"""
    u = np.asarray(u, dtype=np.int8)
    G = _build_generator_matrix(len(u))
    return (u @ G) % 2
