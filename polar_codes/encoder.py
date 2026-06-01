"""
极化码编码器
编码：x = u * G_N，G_N = B_N * F^{\\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 1 << n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形：u[i] <- u[i] XOR u[i+step]，共 log2(N) 层，最后比特倒序。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("Length must be power of 2")

    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    r"""构造 G_N = B_N F^{\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F) % 2
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2
