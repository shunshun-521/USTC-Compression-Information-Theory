"""
极化码编码器
编码：c = F^{\otimes n} @ u（GF(2)），蝶形 XOR 实现 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（非系统化，Arikan 核 F=[[1,1],[0,1]]）。

    蝶形结构：按块大小 N, N/2, ..., 2 依次将右半块 XOR 到左半块。
    """
    u = np.array(u, dtype=int).copy()
    n_len = len(u)
    block = n_len
    while block > 1:
        half = block // 2
        for base in range(0, n_len, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half
    return u
