"""
极化码编码器
编码 x = u G_N，蝶形 XOR：x[j] ^= x[j+step]（与标准 Arikan G 一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """O(N log N) 极化编码。"""
    x = np.array(u, dtype=np.int8).copy()
    n = int(np.log2(len(x)))
    step = 1
    for _ in range(n):
        for i in range(0, len(x), 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    return x.astype(int)


def build_generator_matrix(N):
    """G_N = F^{\\otimes n}，F=[[1,1],[0,1]]（用于校验）。"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(F, G)
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码（校验用）。"""
    N = len(u)
    return (build_generator_matrix(N) @ u) % 2
