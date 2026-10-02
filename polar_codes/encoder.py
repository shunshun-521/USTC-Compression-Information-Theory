"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def bit_reversed_index(i, n):
    """单 index 的 bit-reversal（与 Arikan B_N 一致）"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码：对 u 做蝶形 XOR（等价于 u @ F^{\otimes n}）。
    信道端码字为 u_encoded；与置换 SC 译码器配套使用。
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def polar_encode_with_reversal(u):
    """蝶形编码后再做比特倒序置换（规范形式 u @ B_N @ F^{\otimes n}）。"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(u))
    return x[rev]
