"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    输出为 u @ F^{\\otimes n}；比特倒序由 Permuted SCD 译码器的相位顺序吸收。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("u length must be a power of 2")

    n = int(np.log2(N))
    stage_len = N
    while stage_len > 1:
        half = stage_len // 2
        for p in range(0, N, stage_len):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        stage_len = half

    return u


def build_generator_matrix(N):
    """构造 F^{\\otimes n}（蝶形编码生成矩阵，用于验证）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    A = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        A = np.kron(A, F)
    return A


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print("u =", u)
    print("x (butterfly+BR) =", x)
    print("u @ G =", (u @ G) % 2)
