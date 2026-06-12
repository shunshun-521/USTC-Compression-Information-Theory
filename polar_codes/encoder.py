"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 G_N = F^{\\otimes n} 一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    n = int(np.log2(N))

    for stage in range(1, n + 1):
        block = 1 << stage
        half = block >> 1
        for j in range(0, N, block):
            for i in range(half):
                x[j + i] ^= x[j + i + half]

    return x


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G
