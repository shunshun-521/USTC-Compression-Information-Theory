"""
极化码编码器
编码：x = u * F^⊗n，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev = 0
        for b in range(n):
            if i & (1 << b):
                rev |= 1 << (n - 1 - b)
        indices[i] = rev
    return indices


def polar_encode(u):
    """
    极化码编码（蝶形结构，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block_size = N
    while block_size > 1:
        half = block_size // 2
        for start in range(0, N, block_size):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block_size = half

    return u.astype(int)
