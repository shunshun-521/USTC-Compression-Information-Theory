"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N)
注：比特倒序置换 B_N 由译码端信道 LLR 索引约定统一处理；可用 bit_reversal_permutation 显式重排。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（蝶形 XOR）。
    apply_bit_reversal: 若为 True，对码字做 B_N 置换（x'[i]=x[rev(i)]）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1

    if apply_bit_reversal:
        rev = bit_reversal_permutation(N)
        return u[rev]
    return u
