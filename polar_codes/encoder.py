"""
极化码编码器
编码：x = u * F^{\\otimes n}（蝶形 XOR），与 Permuted SCD 译码器配套
"""
import numpy as np


def _bit_rev_indices(N):
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_rev_indices(N)


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}（与蝶形编码一致）"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


def polar_encode(u):
    """
    极化码编码：O(N log N) 蝶形结构。
    每层对左半部分累加右半部分（模 2）：u[l] ^= u[l + step]。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_block = N
    n = int(np.log2(N))
    for _ in range(n):
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n_block = n_split
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵乘法编码，用于校验蝶形实现"""
    u = np.asarray(u, dtype=int)
    G = polar_generator_matrix(len(u))
    return (u @ G) % 2
