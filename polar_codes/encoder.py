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
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    x = np.zeros(N, dtype=int)
    x[br] = u
    return x


def polar_generator_matrix(N):
    """生成极化码生成矩阵 G_N = B_N F^{⊗n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, br[i]] = 1
    return B @ G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("u:", u)
    print("butterfly+br:", x)
    print("matrix:", x_mat)
    print("match:", np.array_equal(x, x_mat))
