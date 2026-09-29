"""
极化码编码器
编码：x = bitrev(butterfly(u))，与 Permuted SCD 配套
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx & 1) << (n - 1))
    for i in range(1, n):
        rev |= ((idx >> i) & 1) << (n - 1 - i)
    return rev


def _butterfly(u):
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def polar_encode(u):
    """
    极化码编码：先 Arikan 蝶形，再比特倒序置换后发送。
    """
    v = _butterfly(u)
    return v[bit_reversal_permutation(len(u))]


def polar_decode_source(v):
    """蝶形逆（自逆）从 v 恢复 u"""
    return _butterfly(v)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
