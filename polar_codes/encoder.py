"""
极化码编码器
编码：u 经 Kronecker 结构 F^{⊗n} 变换（O(N log N) 蝶形/分块 XOR）
与 SC 译码器采用相同的比特相位约定（译码按比特倒序相位进行）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if (i & (1 << b)):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（分块 XOR，等价于 u · F^{⊗n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（信道映射顺序）
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    size = N
    while size > 1:
        n_split = size // 2
        for p in range(0, N, size):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        size = n_split
    return u
