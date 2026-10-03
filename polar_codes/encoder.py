"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}（Arikan 核 [[1,1],[0,1]]）
"""
import numpy as np

_GEN_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    return int(f"{x:0{n}b}"[::-1], 2)


def _build_generator_matrix(N):
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        u = np.zeros(N, dtype=np.int8)
        u[i] = 1
        G[i] = polar_encode_butterfly(u)
    return G


def polar_encode(u):
    """
    极化码编码：x = u @ G_N（GF(2)）
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    if N not in _GEN_CACHE:
        _GEN_CACHE[N] = _build_generator_matrix(N)
    G = _GEN_CACHE[N]
    return (u @ G) % 2


def polar_encode_butterfly(u):
    """蝶形编码（与 polar_encode 等价）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_block = N
    for _ in range(int(np.log2(N))):
        if n_block == 1:
            break
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n_block = n_split
    return u
