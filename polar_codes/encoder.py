"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码。

    蝶形 XOR 后可选比特倒序置换（与 SC 译码的比特顺序约定一致时可关闭倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    if apply_bit_reversal:
        rev = bit_reversal_permutation(N)
        return u[rev]
    return u


def polar_encode_matrix(u):
    """矩阵乘法参考：x = u * B_N * F^{\\otimes n}"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    G %= 2
    B = np.eye(N, dtype=int)[bit_reversal_permutation(N)]
    GN = (B @ G) % 2
    return (np.asarray(u) @ GN) % 2
