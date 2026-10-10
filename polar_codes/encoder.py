"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与置换 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（x = u * F^{⊗ n}）
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")

    n_stage = N
    for _ in range(n):
        if n_stage == 1:
            break
        n_split = n_stage // 2
        for p in range(0, N, n_stage):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n_stage = n_split

    return u.astype(int)


def polar_encode_matrix(u):
    """GF(2) 矩阵乘法编码，用于校验。"""
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
