"""
极化码编码器
编码：x = u * F^{\\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        x = i
        r = 0
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无比特倒序）。
    与 F^{\\otimes n} 左乘 u 等价（Arikan 核 [[1,1],[0,1]]）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                l = p + k
                u[l] = (u[l] + u[l + half]) % 2
        block = half

    return u


def polar_generator_matrix(N):
    """生成 F^{\\otimes n}（GF(2)）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    n = int(np.log2(N))
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2
