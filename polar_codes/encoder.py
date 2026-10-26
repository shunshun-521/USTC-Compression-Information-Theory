"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return rev.astype(int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    x = np.array(u, dtype=int, copy=True)
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        span = step << 1
        for start in range(0, N, span):
            left = slice(start, start + step)
            right = slice(start + step, start + span)
            x[left] = (x[left] ^ x[right]) & 1
    br = bit_reversal_permutation(N)
    return x[br]


def build_generator_matrix(N):
    """调试用：显式生成矩阵 G_N = B_N F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    B[br, np.arange(N)] = 1
    return (B @ G) % 2
