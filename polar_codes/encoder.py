"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Aff3ct / 5G NR 同款蝶形）：从高位 k=N/2 到 1，x[j+i] ^= x[j+i+k]。
    复杂度 O(N log N)。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    n = len(x)
    k = n // 2
    while k > 0:
        for j in range(0, n, 2 * k):
            for i in range(k):
                x[j + i] ^= x[j + i + k]
        k >>= 1
    return x.astype(int)


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ G) % 2


def polar_encode_matrix(u):
    """矩阵乘法编码 u @ G_N mod 2"""
    N = len(u)
    G = polar_generator_matrix(N)
    return (np.asarray(u, dtype=int) @ G) % 2
