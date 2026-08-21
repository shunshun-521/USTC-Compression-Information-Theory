"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _generator_matrix(n):
    if n not in _G_CACHE:
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        G = F.copy()
        for _ in range(n - 1):
            G = np.kron(G, F)
        _G_CACHE[n] = G.astype(np.int8)
    return _G_CACHE[n]


def polar_encode(u):
    """
    极化码编码：v = u G，x = v[B_N]（比特倒序置换）
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    G = _generator_matrix(n)
    v = (u @ G) % 2
    return v[bit_reversal_permutation(N)]
