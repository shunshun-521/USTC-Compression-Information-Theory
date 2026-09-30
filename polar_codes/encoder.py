"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗ n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def bit_reversed_index(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def _butterfly_xor(u):
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """极化码编码（蝶形 XOR）"""
    return _butterfly_xor(u).astype(int)


def polar_encode_matrix(u):
    """用生成矩阵 G_N = F^{⊗ n} 编码（校验用）"""
    N = len(u)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F) % 2
    return (np.asarray(u, dtype=int) @ G) % 2
