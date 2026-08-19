"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)
    perm = np.empty(N, dtype=int)
    perm[rev] = indices
    return perm


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[j+step] ^= u[j]
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    stage_len = N
    while stage_len > 1:
        half = stage_len // 2
        for base in range(0, N, stage_len):
            for j in range(half):
                u[base + j] ^= u[base + j + half]
        stage_len = half

    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{⊗n}（用于验证）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    m = int(np.log2(N)) - 1
    for _ in range(m):
        G = np.kron(G, F) % 2

    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, br[i]] = 1
    return (B @ G) % 2
