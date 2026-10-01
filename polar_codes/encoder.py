"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def _build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}，行向量 u 满足 x = (u @ G) % 2"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return G % 2


def polar_encode(u):
    """
    极化码编码（与标准蝶形 SC 译码器配套，无额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        n = half
    return u.astype(int)


def polar_encode_with_bitrev(u):
    """含比特倒序置换的编码（G_N = B_N F^{\\otimes n}）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_encode_matrix(u):
    """矩阵乘法编码（用于校验）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = _build_generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode", polar_encode(u), "bitrev", polar_encode_with_bitrev(u))
