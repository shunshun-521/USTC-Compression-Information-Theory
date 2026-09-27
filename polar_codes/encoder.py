"""
极化码编码器
编码：x = F^{\\otimes n} @ u（mod 2），蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def polar_encode(u):
    """
    极化码编码（与 Arikan / 标准蝶形一致：u[l] ^= u[l+step]）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                x[idx] = (x[idx] + x[idx + half]) % 2
        block = half
    return x
