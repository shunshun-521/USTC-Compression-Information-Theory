"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 perm，满足 out[i] = in[perm[i]]"""
    n = int(np.log2(N))
    perm = np.arange(N, dtype=int)
    rev = ((perm[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev[:, ::-1]
    weights = 1 << np.arange(n)
    return (rev * weights).sum(axis=1)


def polar_encode(u):
    """
    极化码编码：蝶形结构（与 F^{⊗n} 生成矩阵一致，信道序码字）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    stage = N
    while stage > 1:
        half = stage // 2
        for base in range(0, N, stage):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        stage = half
    return u.astype(int)


def polar_encode_matrix(u):
    """GF(2) 生成矩阵编码，用于校验"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=int)
    perm = bit_reversal_permutation(N)
    for i, p in enumerate(perm):
        B[i, p] = 1
    return (u @ G) % 2
