"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 rev，满足 out[i] = in[rev[i]]"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def _build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n} over GF(2)"""
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = np.array([[1]], dtype=np.uint8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.uint8)
    for i, j in enumerate(rev):
        B[i, j] = 1
    return (B @ G) % 2


def polar_encode(u):
    """
    极化码非递归编码（蝶形 XOR 至左半区），与 polarcodes / 5G 参考实现一致。
    """
    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    x = u.copy()
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                x[l] ^= x[l + n_split]
        n = n_split
    return x


def polar_encode_matrix(u):
    """矩阵乘法编码（用于校验）"""
    N = len(u)
    G = _build_generator_matrix(N)
    return (u.astype(np.uint8) @ G) % 2
