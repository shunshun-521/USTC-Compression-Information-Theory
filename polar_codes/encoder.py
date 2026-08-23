"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形编码：u[j] ^= u[j + step]"""
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    N = len(u)

    for stage in range(n):
        step = 1 << stage
        for left in range(0, N, 2 * step):
            right = left + step
            u[left:left + step] ^= u[right:right + step]

    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[i] ^= u[i + step]
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = _butterfly_encode(u)
    N = len(u)
    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_encode_no_reversal(u):
    """蝶形编码，不做比特倒序（供译码器内部一致性校验使用）。"""
    return _butterfly_encode(u)
