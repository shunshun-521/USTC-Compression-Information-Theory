"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_kernel_power(N):
    """返回 F^{\\otimes n}（用于 SC 译码 u = x F）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=np.int8).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            x[i : i + step] ^= x[i + step : i + 2 * step]
        step *= 2
    br = bit_reversal_permutation(N)
    return x[br]


def polar_generator_matrix(N):
    """生成矩阵 G_N = B_N F^{\\otimes n}（行向量编码 x = u @ G）。"""
    G = polar_kernel_power(N)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly:", x, "matrix:", x_mat)
