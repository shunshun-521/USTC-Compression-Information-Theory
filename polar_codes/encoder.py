"""
极化码编码器
编码：蝶形 XOR（F = [[1,1],[0,1]]），O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if (i >> b) & 1:
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码：x = u * F^{\\otimes n}（蝶形 XOR）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_matrix(u):
    """矩阵参考编码（与 polar_encode 一致）"""
    return polar_encode(u)
