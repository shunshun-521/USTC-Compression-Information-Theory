"""
极化码编码器
编码：x = u * G_N，G_N = F^⊗n
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def _generator_matrix(n):
    g = np.array([[1, 0], [1, 1]], dtype=int)
    gn = np.array([[1]], dtype=int)
    for _ in range(n):
        gn = np.kron(gn, g)
    return gn


def polar_encode(u):
    """
    极化码编码：x = u * G_N (mod 2)。
    """
    u = np.array(u, dtype=np.int8)
    n = int(np.log2(len(u)))
    g = _generator_matrix(n)
    return (u @ g) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
