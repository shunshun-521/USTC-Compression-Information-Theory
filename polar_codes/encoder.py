"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = index whose bit-reversal is i."""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(np.int64)
    rev = rev * (2 ** np.arange(n - 1, -1, -1))
    return rev.sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = u.shape[0]
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u.astype(np.int8)


def build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n} over GF(2), row u gives codeword u @ G."""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


def polar_encode_matrix(u):
    """Matrix form for verification."""
    u = np.asarray(u, dtype=np.int8)
    G = build_generator_matrix(N=len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("butterfly:", polar_encode(u))
    print("matrix:   ", polar_encode_matrix(u))
