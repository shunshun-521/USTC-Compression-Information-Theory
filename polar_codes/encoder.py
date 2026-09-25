"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ F^{\\otimes n} (mod 2)
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("Length must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                x[idx] ^= x[idx + half]
        block = half
    return x


def polar_encode_matrix(N):
    """生成 F^{\\otimes n}，用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    return G % 2
