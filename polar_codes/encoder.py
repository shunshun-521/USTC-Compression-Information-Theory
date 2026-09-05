"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev[:, ::-1]
    powers = 1 << np.arange(n)
    return (rev * powers).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：每层对所有块执行 u[j] ^= u[j+step]（j 为块内左半索引）
    最后做比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """构造 G_N，满足 x = u @ G_N（mod 2）"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[i, :] = polar_encode(e)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly:", x)
    print("matrix:   ", x_mat)
    assert np.array_equal(x, x_mat), "编码器与生成矩阵不一致"
