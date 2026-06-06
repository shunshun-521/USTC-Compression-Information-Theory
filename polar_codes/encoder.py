"""
极化码编码器
编码：x = u * G_N，利用递归偶/奇结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（递归偶/奇蝶形结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int)
    if len(u) == 1:
        return u.copy()

    N = len(u)
    x = np.zeros(N, dtype=int)
    x[: N // 2] = polar_encode((u[0::2] + u[1::2]) % 2)
    x[N // 2 :] = polar_encode(u[1::2])
    return x
