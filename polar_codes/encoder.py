"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR 结构，等价于 u @ F^{⊗n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2
    return u.astype(int)


def polar_encode_with_reversal(u):
    """带比特倒序置换的编码（u @ G_N = u @ B_N @ F^{⊗n}）"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(u))
    return x[rev]
