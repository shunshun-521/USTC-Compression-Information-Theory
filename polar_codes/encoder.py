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
    """对标量索引 x 做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（非递归蝶形结构，无末尾比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def build_generator_matrix(N):
    """构建 G_N = F^{⊗n}（Arikan 核，无比特倒序）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = build_generator_matrix(4)
    print("matrix multiply:", (u @ G) % 2)
