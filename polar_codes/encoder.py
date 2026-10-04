"""
极化码编码器
编码：对 u 做蝶形极化变换（与 Arikan F^{⊗n} 一致），码字即为变换后的 u
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def bit_reversed(i, n):
    """将 i 的 n 位二进制表示比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def polar_encode(u):
    """
    极化码编码（O(N log N) 蝶形，左半累加右半）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（极化变换后的序列）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u.astype(int)
