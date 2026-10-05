"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}（蝶形 XOR），复杂度 O(N log N)
注：与 B_N 重排信息位等价；信道码字为蝶形输出（与 u @ G 一致）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否对蝶形结果做比特倒序置换

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8)
    n = len(u)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("u length must be a power of 2")

    v = u.astype(np.uint8, copy=True)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                v[j] ^= v[j + step]
        step <<= 1

    if apply_bit_reversal:
        br = bit_reversal_permutation(n)
        v = v[br]
    return v.astype(np.int8)
