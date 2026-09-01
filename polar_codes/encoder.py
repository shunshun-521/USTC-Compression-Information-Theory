"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，不在输出端做比特倒序置换）。

    与 Permuted SC 译码器配对：等价于标准实现中编码后做 B_N 置换。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=int, copy=True)
    N = len(x)
    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step //= 2
    return x


def polar_encode_with_reversal(u):
    """标准编码：蝶形 + 比特倒序置换"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]
