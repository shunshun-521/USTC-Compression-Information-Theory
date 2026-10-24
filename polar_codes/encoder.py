"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx & 1) << (n - 1))
    for i in range(1, n):
        rev |= ((idx >> i) & 1) << (n - 1 - i)
    return rev


def polar_encode(u):
    """
    极化码编码：蝶形结构 u * F^{\\otimes n}（与 SC 译码器自然序 LLR 配套）。
    若需含 B_N 的码字序，请对输出再施加 bit_reversal_permutation。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for step in range(n):
        stride = 1 << step
        for i in range(0, N, stride << 1):
            u[i : i + stride] ^= u[i + stride : i + (stride << 1)]
    return u


def polar_generator_matrix(N):
    """GF(2) 生成矩阵 F^{\\otimes n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G
