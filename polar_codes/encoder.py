"""
极化码编码器
编码：x = polar_transform(u)，Arikan 递归极化变换
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """将 i 的 n 位二进制表示倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def polar_encode(u):
    """
    极化码编码（Arikan 递归极化变换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int)
    if len(u) == 1:
        return u.copy()
    u1u2 = u[::2] ^ u[1::2]
    u2 = u[1::2]
    return np.concatenate([polar_encode(u1u2), polar_encode(u2)])
