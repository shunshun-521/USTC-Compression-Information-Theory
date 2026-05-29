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
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()

    for stage in range(n):
        step = 1 << stage
        for start in range(0, N, 2 * step):
            for i in range(start, start + step):
                x[i] ^= x[i + step]

    brp = bit_reversal_permutation(N)
    return x[brp]


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}，用于验证"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    brp = bit_reversal_permutation(N)
    return G[brp, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = build_generator_matrix(4)
    x_mat = u @ G % 2
    print("matrix encode:", x_mat)
