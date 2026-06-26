"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def butterfly_encode(u):
    """Arikan 蝶形编码（不含比特倒序）"""
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 2 ** (n - layer - 1)
        for j in range(2 ** layer):
            base = 2 * step * j
            for i in range(step):
                u[base + i] ^= u[base + i + step]
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = butterfly_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{⊗n}，用于验证"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return G


def polar_encode_matrix(u):
    """基于生成矩阵的参考编码"""
    u = np.array(u, dtype=int)
    G = build_generator_matrix(len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    print("u:", u)
    print("polar_encode:", x)
    print("matrix encode:", x_ref)
    print("match:", np.array_equal(x, x_ref))
