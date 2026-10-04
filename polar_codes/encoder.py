"""
极化码编码器
编码：x = u * G_N，G_N = F^⊗n（与 SC/SCL 译码器一致）
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def _generator_matrix(N):
    if N in _G_CACHE:
        return _G_CACHE[N]
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F
    while G.shape[0] < N:
        G = np.kron(G, F)
    G = G.astype(np.int8) % 2
    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，x = u @ G_N (mod 2)
    """
    u = np.asarray(u, dtype=np.int8).ravel()
    N = len(u)
    G = _generator_matrix(N)
    return (u @ G) % 2
