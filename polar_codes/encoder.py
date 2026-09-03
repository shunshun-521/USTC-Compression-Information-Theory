"""
极化码编码器
编码：x = u * G_N，蝶形 XOR 结构 O(N log N)
G_N = F^{\\otimes n}，F = [[1,1],[0,1]]（与 Permuted SCD 译码器配套）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for j in range(n):
            r = (r << 1) | ((i >> j) & 1)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无输出比特倒序）。
    与 Permuted SCD 译码器及 G_N = F^{\\otimes n} 一致。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                x[p + k] ^= x[p + k + half]
        block = half
    return x


def polar_generator_matrix(N):
    """生成极化码生成矩阵 G_N = F^{\\otimes n}（GF(2)），F=[[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(F, G)
    return G
