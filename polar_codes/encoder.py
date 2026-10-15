"""
极化码编码器
编码：x = u F^⊗n（Arikan 极化变换），蝶形实现 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=np.int64)
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
    极化码编码（Arikan 变换 u -> x = u G_N，G_N = F^⊗n）。

    蝶形：对每层 s，u[j] ^= u[j + step]（j 为块内上支路索引）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << s
        for i in range(0, N, step * 2):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
