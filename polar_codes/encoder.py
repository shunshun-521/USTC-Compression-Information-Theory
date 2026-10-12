"""
极化码编码器
编码：x = u * G_N，G_N = F^{\otimes n}（Arikan 生成矩阵，无比特倒序）
"""
import numpy as np


def _generator_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组（构造/报告用）。"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u G_N (mod 2)。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = _generator_matrix(N)
    return (u @ G) % 2


def verify_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
