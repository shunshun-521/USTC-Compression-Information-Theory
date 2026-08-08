"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """对标量索引做比特倒序"""
    return int(format(i, f'0{n}b')[::-1], 2)


def polar_encode(u):
    """
    极化码编码（蝶形结构，无显式比特倒序；译码器在比特倒序索引下工作）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N

    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_with_br(u):
    """编码后再做比特倒序置换（部分文献约定）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]
