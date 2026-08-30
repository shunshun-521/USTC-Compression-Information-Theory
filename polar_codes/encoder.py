"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码。

    蝶形结构：每层 (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    等价于 x = u @ G_N，G_N = B_N F^{⊗n}（B_N 为行比特倒序置换）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i: i + step] ^= u[i + step: i + 2 * step]
        step *= 2
    return u


def build_generator_matrix(N):
    """构造 F^{⊗n}（与蝶形编码器一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


def polar_encode_matrix(u):
    """矩阵乘法编码（验证用）"""
    u = np.asarray(u, dtype=np.int8)
    G = build_generator_matrix(len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:   ", x_mat)
    assert np.array_equal(x, x_mat), f"编码器不一致: {x} vs {x_mat}"
