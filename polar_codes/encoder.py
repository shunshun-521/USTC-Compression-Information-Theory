"""
极化码编码器
编码：蝶形 XOR 结构，等价于 polarcodes 非系统编码
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，O(N log N)）。
    与 Permuted SCD 译码器配套使用。
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


def build_generator_matrix(N):
    """Arikan 生成矩阵 F^{\\otimes n}（polarcodes 约定）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(F, G)
    return G % 2
