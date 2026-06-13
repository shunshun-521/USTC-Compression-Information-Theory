"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    for layer in range(n):
        step = 2 ** (layer + 1)
        half = step // 2
        for i in range(0, N, step):
            for j in range(half):
                x[i + j] ^= x[i + j + half]
    brp = bit_reversal_permutation(N)
    return x[brp]


def bit_reverse_llr(llr):
    """将信道 LLR 按比特倒序重排，与编码器输出顺序对齐"""
    N = len(llr)
    brp = bit_reversal_permutation(N)
    return llr[brp]
