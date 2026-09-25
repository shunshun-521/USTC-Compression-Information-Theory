"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形编码（不含比特倒序）"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    size = N
    for _ in range(n):
        split = size // 2
        for p in range(0, N, size):
            for k in range(split):
                u[p + k] ^= u[p + k + split]
        size = split
    return u


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}^T"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    G = G.T
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=int)
    br = bit_reversal_permutation(len(u))
    encoded = _butterfly_encode(u[br])
    return encoded


def polar_encode_matrix(u):
    """矩阵乘法编码 x = G_N @ u"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = build_generator_matrix(N)
    return (G @ u) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:   ", xm)
    assert np.array_equal(x, xm), "butterfly != matrix"
    print("encoder OK")
