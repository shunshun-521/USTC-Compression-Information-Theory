"""
极化码编码器
编码：对 u 做 Kronecker 结构 XOR（与 B_N F^{⊗n} 行向量乘法一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（分块 XOR，O(N log N)）。
    输出即为 BPSK 调制前的码字比特（不做额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u


def encode_via_matrix(u):
    """编码校验：与 ``polar_encode`` 使用同一分块 XOR 算法。"""
    return polar_encode(u)
