"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    蝶形结构：对 step = 1, 2, ..., N/2
        u[i] ^= u[i + step]（i = 0, step, 2*step, ...）
    等价于 x = u @ G_N（GF(2)），G_N = B_N F^{\\otimes n}
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    return u


def build_generator_matrix(N):
    """构建 GF(2) 生成矩阵 G_N，满足 x = G_N @ u（列向量）"""
    G = np.zeros((N, N), dtype=np.int8)
    for j in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[j] = 1
        G[:, j] = polar_encode(e)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (G @ u) % 2
    print("butterfly:", x)
    print("matrix multiply:", x_mat)
    print("match:", np.array_equal(x, x_mat))
