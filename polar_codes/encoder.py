"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        rev[i] = int(bin(i)[2:].zfill(n)[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，x = u * F^{⊗n}）。
    与 G_N = F^{⊗n} 的矩阵乘法等价，SC/BP 译码器与此编码配套。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def polar_encode_with_bitrev(u):
    """含比特倒序置换的编码（x = u * B_N * F^{⊗n}）。"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]


def polar_generator_matrix(N):
    """生成 G_N = F^{⊗n}"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly encode:", x)
    print("matrix encode:", x_mat)
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
