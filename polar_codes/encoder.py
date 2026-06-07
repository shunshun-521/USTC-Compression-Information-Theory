"""
极化码编码器
编码：x = u * G_N（F^⊗n），利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形左支 XOR，对应 G_N = F^{\otimes n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=int).copy()
    n = int(np.log2(len(x)))
    N = len(x)
    for layer in range(n):
        step = 1 << layer
        for left in range(0, N, 2 * step):
            for i in range(left, left + step):
                x[i] ^= x[i + step]
    return x
