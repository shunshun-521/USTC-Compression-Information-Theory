"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引：out[i] = in[perm[i]]，perm[i]=bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    等价于 c = G_N u（GF(2)），G_N = B_N F^{\\otimes n}。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = N
    while step > 1:
        step //= 2
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]

    perm = bit_reversal_permutation(N)
    x = u[perm]
    return x.astype(np.int8)


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于测试）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F) % 2
    perm = bit_reversal_permutation(N)
    G = G[perm, :]
    return G.astype(np.int8)
