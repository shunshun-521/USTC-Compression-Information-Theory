"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序。"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    算法：先将 u 散布到比特倒序位置，再执行 log2(N) 层蝶形 XOR。
    等价于 x = u @ (B_N F^{\\otimes n})。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    br = bit_reversal_permutation(N)
    d = np.zeros(N, dtype=int)
    d[br] = u

    for step in range(n):
        stride = 1 << step
        for start in range(0, N, 2 * stride):
            d[start : start + stride] ^= d[start + stride : start + 2 * stride]

    return d


def build_generator_matrix(N):
    """构建 GF(2) 生成矩阵 G_N = B_N F^{\\otimes n}，用于验证。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return G % 2
