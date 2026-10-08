"""
极化码编码器
编码：x = u F^{\\otimes n}（GF(2)），与 SC/SCL 译码器因子图一致
"""
import numpy as np

_G_CACHE = {}


def _bit_rev_indices(N):
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_rev_indices(N)


def polar_encode(u):
    """
    极化码编码（Kronecker F^{\\otimes n} 蝶形，无额外比特倒序）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    length = x.size
    if length & (length - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x


def polar_generator_matrix(N):
    """由编码器定义的生成矩阵（行 i 为 encode(e_i)）。"""
    if N not in _G_CACHE:
        G = np.zeros((N, N), dtype=int)
        for i in range(N):
            e = np.zeros(N, dtype=int)
            e[i] = 1
            G[i] = polar_encode(e)
        _G_CACHE[N] = G
    return _G_CACHE[N].copy()


def build_generator_from_encoder(N):
    return polar_generator_matrix(N)
