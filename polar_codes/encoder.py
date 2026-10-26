"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = index whose bit-reversal is i"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def butterfly_encode(u):
    """蝶形 XOR（与 aff3ct Encoder_polar::light_encode 一致）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    k = N // 2
    while k > 0:
        for j in range(0, N, 2 * k):
            for i in range(k):
                u[j + i] ^= u[j + i + k]
        k >>= 1
    return u


def polar_encode(u):
    """
    极化码编码（蝶形 + 输出比特倒序置换）。
    """
    x = butterfly_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br].astype(int)


def polar_encode_matrix(u):
    """GF(2) 矩阵编码，用于验证"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    Gn = (np.eye(N, dtype=int)[br] @ G) % 2
    return (u @ Gn) % 2
