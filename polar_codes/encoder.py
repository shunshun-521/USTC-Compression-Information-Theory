"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    x = np.array(u, dtype=np.int8).copy()
    N = len(x)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                x[a] ^= x[b]
        step *= 2
    return x


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    x = _butterfly_encode(u)
    return x[bit_reversal_permutation(len(x))]


def polar_encode_natural(u):
    """仅蝶形变换，不做比特倒序（用于构造/调试）。"""
    return _butterfly_encode(u)
