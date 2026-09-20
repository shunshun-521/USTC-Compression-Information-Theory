"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N) 复杂度（无输出比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    n_split = N
    for _ in range(n):
        n_split //= 2
        for p in range(0, N, n_split * 2):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
    return u
