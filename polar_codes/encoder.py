"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    return int(format(i, f'0{n}b')[::-1], 2)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构与比特倒序置换共同实现 G_N = B_N F^{\\otimes n}。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n_block = N
    while n_block > 1:
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n_block = n_split
    rev = bit_reversal_permutation(N)
    return u[rev]


def build_generator_matrix(N):
    """构建生成矩阵 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[rev]
    return (B @ F_n) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print("u =", u)
    print("x =", x)
    print("matrix encode:", (u @ G) % 2)
