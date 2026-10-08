"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = sum(((idx >> k) & 1) << (n - 1 - k) for k in range(n))
    return rev.astype(np.int64)


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
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    # 比特倒序置换：与 Permuted SCD 配合，码字为蝶形输出经 B_N 置换
    rev = bit_reversal_permutation(N)
    x = u.copy()
    return x[rev]


def build_generator_matrix(N):
    """显式构造 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[rev, :]


def polar_encode_matrix(u):
    """矩阵乘法编码（校验用）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
