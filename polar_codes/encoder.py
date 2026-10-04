"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))

    def reverse_bits(i):
        r = 0
        for _ in range(n):
            r = (r << 1) | (i & 1)
            i >>= 1
        return r

    return np.array([reverse_bits(i) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单 index 比特倒序（与 mcba1n polar-codes 一致）"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（非递归，大 block 优先，与标准 G_N = F^{⊗n} 一致，无输出比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u
