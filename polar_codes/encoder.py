"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：u[i] ^= u[i+step]，共 log2(N) 层，最后 x[i] = v[bit_reverse(i)]。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step <<= 1

    br = bit_reversal_permutation(N)
    return u[br].astype(np.int8)


def build_generator_matrix(N):
    """Arikan 核 F=[[1,1],[0,1]] 的 n 次 Kronecker积，再经 B_N 行置换。"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


def polar_encode_matrix(u):
    """u @ G_N mod 2"""
    u = np.asarray(u, dtype=np.int8)
    G = build_generator_matrix(len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:", xm)
