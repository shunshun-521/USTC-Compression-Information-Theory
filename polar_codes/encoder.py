"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u * B_N * F^{\\otimes n}（3GPP 蝶形 + 比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for i in range(n):
        step = 2**i
        for j in range(step):
            for k in range(2 ** (n - i - 1)):
                idx = k * 2 * step + j
                u[idx] ^= u[idx + step]
    return u[bit_reversal_permutation(N)]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    GN = (B @ G) % 2
    return (u @ GN) % 2
