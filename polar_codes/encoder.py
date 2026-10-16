"""
极化码编码器
编码：蝶形 XOR（与 u @ F^{\\otimes n} 等价）
"""
import numpy as np


def bit_reversal_permutation(N):
    n = int(np.log2(N))
    return np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（非递归蝶形，O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def build_generator_matrix(N):
    """G = F^{\\otimes n}（列向量 x = G u）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int64)
    G = np.array([[1]], dtype=np.int64)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print("encode:", x)
    print("u@G:", (u @ G) % 2)
    print("G@u:", (G @ u) % 2)
