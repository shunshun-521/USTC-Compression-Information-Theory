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
    极化码编码（mcba1n 风格，无显式比特倒序；译码器按倒序处理）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8, copy=True)
    n = int(np.log2(len(u)))
    block = len(u)
    while block > 1:
        half = block // 2
        for p in range(0, len(u), block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half
    return u
