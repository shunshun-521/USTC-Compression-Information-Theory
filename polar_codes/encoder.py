"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)


def _bit_reversed(i, n):
    return int(format(i, f'0{n}b')[::-1], 2)


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

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]

    br = bit_reversal_permutation(N)
    x = u[br]
    return x


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    G = np.array([[1]], dtype=np.int8)
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode test:", x)
    G = build_generator_matrix(4)
    x_mat = np.dot(u, G) % 2
    print("matrix encode:", x_mat)
