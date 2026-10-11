"""
极化码编码器
等价于 G_N 的 stage-wise XOR 编码（与蝶形 + 比特倒序相同码本）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def _gen_gather_indices(n):
    """Sionna 风格的逐层 gather 索引（用于 XOR 编码）。"""
    N = 1 << n
    ind_gather = np.ones((n, N + 1), dtype=np.int32) * N
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码。
    参数 u 为长度 N 的源向量（含冻结位，冻结位应为 0）。
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    n = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_gather_indices(n)
    for s in range(n):
        ind_helper = ind_gather[s]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(int)


def polar_encode_info(info_bits, info_indices, N):
    """将 K 个信息比特填入长度 N 的源向量并编码。"""
    u = np.zeros(N, dtype=int)
    u[np.asarray(info_indices, dtype=int)] = info_bits
    return polar_encode(u)
