"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx & 1) << (n - 1))
    for i in range(1, n):
        rev |= ((idx >> i) & 1) << (n - 1 - i)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if N != 2 ** n:
        raise ValueError("u length must be power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x.astype(int)


def polar_generator_matrix(N):
    """G_N = B_N F^{\\otimes n} over GF(2)，用于校验"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    G = G[rev, :]
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u, "polar_encode:", x, "matrix:", x_ref)
