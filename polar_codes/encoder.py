"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}（标准 Arikan 生成矩阵）
利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    """构建 G_N = F^{⊗ n}"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码：x = u @ G_N (mod 2)，蝶形实现。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2


def polar_encode_butterfly(u):
    """蝶形递归编码（与矩阵乘法等价）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for m in range(n):
        step = 1 << m
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x1 = polar_encode(u)
    x2 = polar_encode_butterfly(u)
    print("u:", u)
    print("matrix:", x1)
    print("butterfly:", x2)
