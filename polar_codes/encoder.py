"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（蝶形递归结构，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=np.int8).copy()
    N = len(x)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                x[idx] ^= x[idx + half]
        block = half
    return x


def polar_encode_with_bit_reversal(u):
    """带输出比特倒序的编码（部分文献约定）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]
