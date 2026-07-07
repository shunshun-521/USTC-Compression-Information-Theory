"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构与标准极化码一致；末尾比特倒序置换与 Permuted SCD 译码器配套。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] = u[j] ^ u[j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """构建生成矩阵 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return G


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("matrix encode:", x_mat)
