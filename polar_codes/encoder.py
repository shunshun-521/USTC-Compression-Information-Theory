"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def polar_generator_matrix(N):
    """返回 GF(2) 上的极化生成矩阵 G_N = F^{\\otimes n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def _butterfly_concat(left, right):
    """极化蝶形：上支为 XOR，下支为原右半。"""
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    upper = left ^ right
    return np.concatenate([upper, right])


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，与 G_N 矩阵乘法等价）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    m = 1
    for _ in range(n):
        for i in range(0, N, 2 * m):
            block = _butterfly_concat(u[i : i + m], u[i + m : i + 2 * m])
            u[i : i + 2 * m] = block
        m *= 2
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵形式编码（用于校验与 BP 早停）。"""
    u = np.asarray(u, dtype=int).ravel()
    G = polar_generator_matrix(len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [1, 1, 0, 1]), f"编码器错误: {x2}"
    assert np.array_equal(polar_encode(u), polar_encode_matrix(u))
    print("encoder OK")
