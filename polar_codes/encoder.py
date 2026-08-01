"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构按块长从 N 递减至 2 执行 XOR，最后做比特倒序置换。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    rev_idx = bit_reversal_permutation(N)
    return u[rev_idx]


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}（GF(2)），用于验证。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    result = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        result = np.kron(result, F) % 2
    rev = bit_reversal_permutation(N)
    return result[rev, :] % 2


def polar_encode_matrix(u):
    """矩阵乘法编码（验证用）。"""
    u = np.array(u, dtype=int)
    G = build_generator_matrix(N=len(u))
    return (u @ G) % 2
