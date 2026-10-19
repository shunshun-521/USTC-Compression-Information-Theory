"""
极化码编码器
编码：x = u * F_N，蝶形 XOR 结构 O(N log N)（与 SC 译码器同一约定，无额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Kronecker F 变换，GF(2)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    length = x.size
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}（无比特倒序行置换）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    return G % 2
