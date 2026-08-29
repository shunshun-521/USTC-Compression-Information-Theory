"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形变换，输出即为信道码字）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    n_block = N
    while n_block > 1:
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n_block = n_split
    return u


def polar_encode_with_br(u):
    """带比特倒序置换的编码（与 polar_encode + BR 信道映射等价）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]
