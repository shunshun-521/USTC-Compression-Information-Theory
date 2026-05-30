"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC/SCL 译码器因子图一致）。

    注：输出为码字自然序；比特倒序由 G_N 中的 B_N 吸收在蝶形层序中。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 1 << layer
        stride = step << 1
        for i in range(0, N, stride):
            for j in range(step):
                a = u[i + j]
                b = u[i + j + step]
                u[i + j] = a ^ b
                u[i + j + step] = b

    return u


def polar_encode_with_reversal(u):
    """含显式比特倒序的编码（与矩阵 G_N = B_N F^{⊗n} 一致）。"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]


def polar_encode_matrix(u):
    """矩阵乘法编码（用于验证）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[rev]
    GN = (B @ G) % 2
    return (u @ GN) % 2
