"""
极化码编码器
编码：x = u * F_N（Kronecker 积，无比特倒序），O(N log N) 蝶形
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in idx:
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（标准 F^{\\otimes n} 变换，与 SC/SCL 译码器一致）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = x.size
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start: start + step] ^= x[start + step: start + 2 * step]
        step <<= 1
    return x.astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G = F^{\\otimes n}（GF(2)）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2
