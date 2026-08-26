"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_rev_indices(N):
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_rev_indices(N)


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 Arikan / 5G 非系统编码一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        n = half
    return u


def polar_encode_with_bitrev(u):
    """带输出比特倒序的编码（部分教材约定）"""
    return polar_encode(u)[_bit_rev_indices(len(u))]


def polar_encode_matrix(N):
    """生成 N x N 生成矩阵 G_N = B_N F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    rev = _bit_rev_indices(N)
    return G[rev, :]
