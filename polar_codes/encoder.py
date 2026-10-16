"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 u @ F^⊗n 等价）。

    信道 LLR 顺序与标准比特序一致，不再额外做比特倒序置换
    （与译码器、BP 早停编码一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N
    for layer in range(n):
        step = 2 ** layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u.astype(int)


def polar_encode_with_bit_reversal(u):
    """带比特倒序置换的编码（部分教材定义 G_N = B_N F^⊗n）。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]
