"""
极化码编码器
编码 x = u * F^{\\otimes n}，O(N log N) 蝶形结构
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，无输出比特倒序）。
    与 u @ F^{\otimes n}（F=[[1,0],[1,1]]）等价。
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    block = 1
    while block < N:
        for start in range(0, N, 2 * block):
            for k in range(block):
                u[start + k] ^= u[start + k + block]
        block *= 2
    return u.astype(int)
