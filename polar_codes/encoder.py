"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    return ((indices & 1) << (n - 1) |
            ((indices >> 1) & 1) << (n - 2) |
            ((indices >> 2) & 1) << (n - 3) |
            ((indices >> 3) & 1) << (n - 4) |
            ((indices >> 4) & 1) << (n - 5) |
            ((indices >> 5) & 1) << (n - 6) |
            ((indices >> 6) & 1) << (n - 7) |
            ((indices >> 7) & 1) << (n - 8) |
            ((indices >> 8) & 1) << (n - 9) |
            ((indices >> 9) & 1) << (n - 10))


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 2 ** layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("matrix encode:", x_mat)
