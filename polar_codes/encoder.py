"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
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
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1
    # 比特倒序置换（与标准 G_N = B_N F^{⊗n} 一致）
    return u[bit_reversal_permutation(N)]


def polar_generator_matrix(N):
    """GF(2) 生成矩阵，用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    m = 1
    while m < N:
        G = np.kron(G, F) % 2
        m <<= 1
    B = np.zeros((N, N), dtype=np.int8)
    rev = bit_reversal_permutation(N)
    for i, r in enumerate(rev):
        B[i, r] = 1
    return (B @ G) % 2
