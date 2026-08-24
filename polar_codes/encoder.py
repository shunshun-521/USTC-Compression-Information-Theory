"""
极化码编码器
编码：x = u * G_N，利用蝶形/gather 结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _gen_gather_indices(N):
    """预计算 Sionna/Arikan 风格的 XOR-gather 编码索引"""
    n = int(np.log2(N))
    ind_gather = np.ones((n, N + 1), dtype=np.int32) * N
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


_GATHER_CACHE = {}


def _get_gather_indices(N):
    if N not in _GATHER_CACHE:
        _GATHER_CACHE[N] = _gen_gather_indices(N)
    return _GATHER_CACHE[N]


def polar_encode(u):
    """
    极化码编码（XOR-gather 蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    ind_gather = _get_gather_indices(N)
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u.astype(np.uint8)

    for s in range(n):
        idx = ind_gather[s, :N]
        x[:N] ^= x[idx]

    return x[:N].astype(np.int8)


def polar_encode_butterfly(u):
    """
    蝶形编码 + 比特倒序（u @ B_N F^{⊗n}，用于对照验证）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]

    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """构建 gather 编码对应的生成矩阵（用于验证）"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[i] = 1
        G[i] = polar_encode(e)
    return G
