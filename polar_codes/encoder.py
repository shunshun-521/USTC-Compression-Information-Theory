"""
极化码编码器
编码：x = G @ u（Arikan 核 F=[[1,1],[0,1]]），蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回比特倒序置换索引：out[i] = in[rev(i)]"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字（G @ u，mod 2）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    block = n
    for _ in range(int(np.log2(n))):
        if block == 1:
            break
        n_split = block // 2
        for p in range(0, n, block):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        block = n_split
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵法编码，用于单元测试"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    return (G @ u) % 2
