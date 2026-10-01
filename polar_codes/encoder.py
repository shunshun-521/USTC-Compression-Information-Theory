"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    assert 2**n == N
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(np.int64)
    rev = rev[:, ::-1]
    powers = 1 << np.arange(n)
    return (rev * powers).sum(axis=1)


def polar_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（行向量为码字生成）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :].astype(np.int8)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 mcba1n/polar-codes 一致，无输出比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_split = N
    while n_split > 1:
        half = n_split // 2
        for p in range(0, N, n_split):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        n_split = half
    return u


def polar_encode_matrix(u):
    """矩阵乘法编码（用于校验）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = polar_generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    xm = polar_encode_matrix(u)
    print("matrix:", xm)
