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
    """对标量索引做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，无输出比特倒序）。
    等价于 x = G_N @ u，G_N = F^{\\otimes n}，F = [[1,1],[0,1]]。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    assert N == 2 ** int(np.log2(N))

    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        n = half
    return u


def build_generator_matrix(N):
    """构建 G_N = F^{\\otimes n}，F = [[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.block([[G, G], [np.zeros_like(G), G]])
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码，用于验证"""
    u = np.asarray(u, dtype=np.int8)
    G = build_generator_matrix(len(u))
    return (G @ u) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:   ", x_mat)
    assert np.array_equal(x, x_mat)
