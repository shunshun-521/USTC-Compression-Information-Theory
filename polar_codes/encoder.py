"""
极化码编码器
编码：x = u * F_N（Kronecker 幂），蝶形 XOR，非比特倒序约定（与 SC 译码器一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（polar / Kronecker 变换）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    length = x.size
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x.astype(int)
