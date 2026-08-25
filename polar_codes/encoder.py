"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1

    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))

    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)

    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, r in enumerate(rev):
        B[i, r] = 1

    G = (B @ F_n) % 2
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:   ", x_mat)
    assert np.array_equal(polar_encode(u), polar_encode_matrix(u))
