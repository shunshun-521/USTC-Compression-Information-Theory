"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def _gen_encode_indices(n):
    """预计算编码阶段的 XOR 索引（与 Sionna PolarEncoder 一致）"""
    nb_stages = int(np.log2(n))
    ind_gather = np.ones((nb_stages, n + 1), dtype=np.int32) * n
    for s in range(nb_stages):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


_ENCODE_CACHE = {}


def polar_encode(u):
    """
    极化码编码。

    将信息位/冻结位向量 u 映射为码字 x（含比特倒序置换，与 GA 构造索引一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    if n not in _ENCODE_CACHE:
        _ENCODE_CACHE[n] = _gen_encode_indices(n)
    ind_gather = _ENCODE_CACHE[n]

    x = np.zeros(n + 1, dtype=np.uint8)
    x[:n] = u
    for s in range(int(np.log2(n))):
        x = np.bitwise_xor(x, x[ind_gather[s, :]])
    x = x[:n]
    return x[bit_reversal_permutation(n)]
