"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{\otimes n}，F = [[1,0],[1,1]]
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码：x = u * G_N（G_N = F^{\otimes n}）。

    蝶形实现：每层 u[k] ^= u[k+step]，复杂度 O(N log N)。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 2 ** layer
        for start in range(0, N, 2 * step):
            for k in range(start, start + step):
                u[k] ^= u[k + step]
    return u
