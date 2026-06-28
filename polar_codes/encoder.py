"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _polar_transform(u):
    """极化变换（与 G_N = B_N F^{\\otimes n} 的 u @ G 一致）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_split = N
    while n_split > 1:
        half = n_split // 2
        for p in range(0, N, n_split):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        n_split //= 2
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = _polar_transform(u)
    br = bit_reversal_permutation(len(u))
    return u[br]


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n} B_N，满足 polar_encode(u) = (u @ G_N) % 2"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (G @ B) % 2
