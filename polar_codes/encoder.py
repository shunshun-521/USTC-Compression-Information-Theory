"""
极化码编码器
编码：x = u * F^{\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否在编码后做 B_N 比特倒序置换

    返回：
        x: 长度为 N 的码字

    蝶形结构：每层 u[i] ^= u[i + step]（右半 XOR 到左半），共 log2(N) 层。
    默认输出 x = u * F^{\otimes n}，与 Vangala 2014 SC 译码器配套。
    若 apply_bit_reversal=True，则 x = u * F^{\otimes n} * B_N。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = N
    while n > 1:
        n_split = n // 2
        for base in range(0, N, n):
            left = slice(base, base + n_split)
            right = slice(base + n_split, base + n_split + n_split)
            u[left] = (u[left] ^ u[right]) & 1
        n = n_split

    if apply_bit_reversal:
        u = u[bit_reversal_permutation(N)]
    return u.astype(int)


def polar_encode_with_bit_reversal(u):
    """x = u * F^{\otimes n} * B_N"""
    return polar_encode(u, apply_bit_reversal=True)
