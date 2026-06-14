"""
极化码编码器
编码：蝶形 XOR，x = u @ F^{\\otimes n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做比特倒序"""
    return int(f"{x:0{n}b}"[::-1], 2)


def polar_encode(u):
    """
    极化码蝶形编码（O(N log N)）。
    与比特倒序相位 SC 译码器配套使用。
    """
    u = np.asarray(u, dtype=np.int32).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}，用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int32)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G[bit_reversal_permutation(N)]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u =", u, "-> x =", polar_encode(u))
