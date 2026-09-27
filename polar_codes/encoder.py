"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_rev_indices(N)


def _bit_rev_indices(N):
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def build_generator_matrix(N):
    """G_N = B_N F^{⊗n}（与 Arıkan 标准一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    br = _bit_rev_indices(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：每层 (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    最后比特倒序置换。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    br = _bit_rev_indices(N)
    return u[br]


def polar_encode_matrix(u):
    """矩阵形式编码，用于校验与 BP 早停重编码"""
    u = np.asarray(u, dtype=int).reshape(-1)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2
