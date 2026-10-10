"""
极化码编码器
编码：x = G_N u（G_N = F^{\\otimes n}），蝶形 O(N log N)
比特倒序置换用于与 B_N F^{\\otimes n} 行向量形式对齐。
"""
import numpy as np


def bit_reversal_permutation(N):
    """out[i] = in[rev(i)] 的索引 rev"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode_core(u):
    """x = G_N @ u，G_N 为 F^{\\otimes n}（下三角）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for j in range(0, N, 2 * step):
            u[j + step:j + 2 * step] = (
                u[j + step:j + 2 * step] + u[j:j + step]
            ) % 2
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    与 x = u @ (B_N G_N) 在 GF(2) 上等价。
    """
    core = polar_encode_core(u)
    brp = bit_reversal_permutation(len(core))
    return core[brp]


def polar_decode_core(x):
    """u = G_N^{-1} x（蝶形逆变换，与 polar_encode_core 互逆）。"""
    x = np.asarray(x, dtype=np.int8).copy()
    N = len(x)
    n = int(np.log2(N))
    for layer in reversed(range(n)):
        step = 1 << layer
        for j in range(0, N, 2 * step):
            x[j + step:j + 2 * step] = (
                x[j + step:j + 2 * step] + x[j:j + step]
            ) % 2
    return x


def generator_matrix(N):
    """G_N = B_N F^{\\otimes n}，满足 polar_encode(u) = (u @ G_N) % 2。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F
    while G.shape[0] < N:
        G = np.kron(G, F)
    brp = bit_reversal_permutation(N)
    return G[brp] % 2
