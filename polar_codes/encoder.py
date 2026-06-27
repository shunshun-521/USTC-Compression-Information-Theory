"""
极化码编码器
编码：u * F_N，蝶形 XOR（与 SC 译码器配套）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = ((indices[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev[:, ::-1].dot(1 << np.arange(n))
    return rev


def polar_encode(u):
    """
    极化码非系统化编码。
    对长度为 N 的源向量 u 施加 F_N 变换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    for level in range(n - 1, -1, -1):
        step = 1 << (n - level - 1)
        groups = 1 << level
        for g in range(groups):
            start = 2 * g * step
            for p in range(step):
                u[p + start] ^= u[p + start + step]
    return u


def polar_generator_matrix(N):
    """生成 F_N = F^{\\otimes n}"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G
