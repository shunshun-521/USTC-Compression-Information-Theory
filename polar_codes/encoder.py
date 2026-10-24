"""
极化码编码器
编码：蝶形 XOR（标准非系统化极化编码），复杂度 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def _bit_reversal_indices(N):
    return bit_reversal_permutation(N)


def polar_encode(u):
    """极化码编码（与标准蝶形 XOR 编码一致）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    for _ in range(int(np.log2(N))):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u.astype(np.int8)


def polar_encode_no_reversal(u):
    return polar_encode(u)
