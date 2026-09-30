"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx & 1) << (n - 1))
    for i in range(1, n):
        rev |= ((idx >> i) & 1) << (n - 1 - i)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for step in range(n):
        stride = 1 << step
        for i in range(0, N, 2 * stride):
            u[i : i + stride] ^= u[i + stride : i + 2 * stride]
    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_generator_matrix(N):
    """生成矩阵 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[rev, :]
