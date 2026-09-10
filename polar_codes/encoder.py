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


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形 XOR 后做比特倒序置换，等价于 x = u @ (B_N F^{⊗n})。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i: i + step] ^= u[i + step: i + 2 * step]
        step *= 2

    br = bit_reversal_permutation(N)
    return u[br]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    GN = (B @ G) % 2
    return (u @ GN) % 2
