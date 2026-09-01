"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构等价于 Arikan 生成矩阵 F^{⊗n}，最后做比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n_stage = N
    for _ in range(int(np.log2(N))):
        if n_stage == 1:
            break
        half = n_stage // 2
        for base in range(0, N, n_stage):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        n_stage = half

    br = bit_reversal_permutation(N)
    return u[br]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))

    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)

    br = bit_reversal_permutation(N)
    # x = u @ G, then bit-reverse: x_br[i] = x[br[i]]
    x = (u @ G) % 2
    return x[br]
