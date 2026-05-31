"""
极化码编码器
编码：u 经蝶形变换（左半累加 XOR）得到码字 x
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序（MSB 在左）"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（非系统化，与 SC 译码器配套）。

    对每个块：u[l] ^= u[l + block/2]（左分支吸收右分支）
    """
    u = np.asarray(u, dtype=int).copy()
    n = len(u)
    if n & (n - 1) != 0:
        raise ValueError("Length must be power of 2")
    block = n
    while block > 1:
        half = block // 2
        for p in range(0, n, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u
