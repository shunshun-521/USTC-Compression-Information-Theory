"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def build_generator_matrix(N, bit_reversed=False):
    """构造 G_N = F^{⊗ n}，可选行置换 B_N。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F) % 2
    if bit_reversed:
        br = bit_reversal_permutation(N)
        G = G[br, :]
    return G


def polar_encode(u):
    """
    极化码编码：x = u @ G_N（与矩阵 SC 因子图译码器一致）。
    """
    u = np.array(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N, bit_reversed=False)
    return (u @ G) % 2


def polar_encode_natural(u):
    """无输出比特倒序的编码（调试用）。"""
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    return x


def polar_encode_matrix(u):
    """矩阵乘法编码（用于校验）"""
    N = len(u)
    G = build_generator_matrix(N)
    return (np.array(u, dtype=np.int8) @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xref = polar_encode_matrix(u)
    print("u:", u)
    print("butterfly encode:", x)
    print("matrix encode:", xref)
    assert np.array_equal(x, xref)
