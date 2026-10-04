"""
极化码编码器
编码：x = u F^{\\otimes n}（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：x = u F^{\\otimes n}（mod 2），与 SC 译码器配套。
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    n = int(np.log2(N))
    for _ in range(n):
        for start in range(0, N, 2 * step):
            x[start:start + step] ^= x[start + step:start + 2 * step]
        step <<= 1
    return x.astype(int)


def polar_f_transform(x):
    """对向量 x 施加 F^{\\otimes n}（与 polar_encode 相同）。"""
    return polar_encode(x)
