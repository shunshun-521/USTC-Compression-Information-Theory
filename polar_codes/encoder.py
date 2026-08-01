"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev = 0
        for b in range(n):
            if (i >> b) & 1:
                rev |= 1 << (n - 1 - b)
        indices[i] = rev
    return indices


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。
    与标准 SC 因子图配套：对 u 做分层 XOR 极化变换得到码字 x。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u
