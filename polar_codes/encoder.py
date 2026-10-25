"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r = (r << 1) | ((i >> b) & 1)
        rev[i] = r
    return rev


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
    for layer in range(1, n + 1):
        step = 2 ** layer
        half = 2 ** (layer - 1)
        for block in range(2 ** (n - layer)):
            base = block * step
            for i in range(half):
                i1 = base + i
                i2 = i1 + half
                u[i1] = (u[i1] + u[i2]) % 2
    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]
