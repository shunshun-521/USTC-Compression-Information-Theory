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
    极化码编码（蝶形结构，与 G_N = F^{\otimes n} 一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    n = int(np.log2(len(u)))

    step = 1
    for _ in range(n):
        for i in range(0, len(u), 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2

    return u


def polar_encode_matrix(u):
    """通过生成矩阵编码，用于验证。"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x (butterfly) =", x)
    print("x (matrix)    =", polar_encode_matrix(u))
    assert np.array_equal(polar_encode(u), polar_encode_matrix(u))
