"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def _butterfly_combine(a, b):
    """蝶形合并：[a XOR b, b]"""
    a = np.asarray(a, dtype=np.int8)
    b = np.asarray(b, dtype=np.int8)
    return np.concatenate([a ^ b, b])


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC 译码器配套）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    m = 1
    for _ in range(n):
        for i in range(0, N, 2 * m):
            block = _butterfly_combine(u[i : i + m], u[i + m : i + 2 * m])
            u[i : i + 2 * m] = block
        m *= 2
    return u
