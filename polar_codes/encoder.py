"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.array(u, dtype=np.int8).copy()
    N = u.size
    if N & (N - 1):
        raise ValueError("N must be power of 2")

    n = int(np.log2(N))
    for step in range(n):
        stride = 1 << (step + 1)
        half = stride >> 1
        for base in range(0, N, stride):
            for i in range(half):
                u[base + i] ^= u[base + i + half]

    return u


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}（与蝶形 XOR 编码一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


def polar_encode_matrix(u):
    u = np.array(u, dtype=np.int8)
    G = polar_generator_matrix(u.size)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("butterfly:", x)
    print("matrix:", polar_encode_matrix(u))
