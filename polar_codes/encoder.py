"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}，F = [[1,0],[1,1]]
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def _generator_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码（矩阵法，与 SC 因子图一致）。
    蝶形等价实现见 polar_encode_butterfly（不含比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8)
    G = _generator_matrix(len(u))
    return (u @ G) % 2


def polar_encode_butterfly(u):
    """O(N log N) 蝶形编码（与 polar_encode 结果一致，无额外比特倒序）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_block = N
    while n_block > 1:
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n_block = n_split
    return u


def polar_encode_matrix(u):
    """与 polar_encode 相同（保留接口）。"""
    return polar_encode(u)
