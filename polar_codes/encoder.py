"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{\\otimes n}，F = [[1,1],[0,1]]（与 SC 因子图一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 Arikan 核 F=[[1,1],[0,1]] 一致）。
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                x[a] ^= x[b]
        step <<= 1
    return x


def polar_generator_matrix(N):
    """生成 G_N = F^{\\otimes n}"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G
