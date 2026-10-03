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
    for i in range(N):
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（与因子图一致的蝶形 XOR，信道端顺序与 u 索引一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    stage_len = N
    while stage_len > 1:
        n_split = stage_len // 2
        for p in range(0, N, stage_len):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        stage_len = n_split
    return u


def polar_encode_matrix(u):
    """矩阵形式编码，用于校验。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"蝶形与矩阵不一致: {x} vs {xm}"
    print("encoder test:", u, "->", x)
