"""
极化码编码器
蝶形 XOR 编码，O(N log N)，与 mcba1n SCD 译码器配套
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码蝶形 XOR 编码（无输出比特倒序；SCD 在比特倒序索引顺序上译码）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    stage_len = N
    for _ in range(n):
        n_split = stage_len // 2
        for p in range(0, N, stage_len):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        stage_len = n_split
    return u
