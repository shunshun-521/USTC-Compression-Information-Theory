"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] >> np.arange(n)) & 1).dot(1 << np.arange(n))
    return rev.astype(np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        span = step << 1
        for start in range(0, N, span):
            left = start
            right = start + step
            u[left:right] ^= u[right : right + step]

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x.astype(int)


def polar_encode_matrix(u):
    """矩阵乘法编码（用于校验蝶形实现）。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    G = G[rev, :]
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    print("matrix:", polar_encode_matrix(u))
