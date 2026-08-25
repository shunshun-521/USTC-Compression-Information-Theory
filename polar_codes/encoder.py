"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u * G_N，G_N = B_N F^{\\otimes n}。

    等价蝶形实现：从细粒度到粗粒度依次执行
    u[idx1] ^= u[idx2]，最后得到码字（与矩阵乘法一致）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        span = step << 1
        for base in range(0, N, span):
            for i in range(step):
                u[base + i] ^= u[base + step + i]
    return u


def polar_generator_matrix(N):
    """生成与蝶形编码一致的 G_N = F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    print("u =", u)
    print("x =", x)
    print("u @ G =", (u @ G) % 2)
