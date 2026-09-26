"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def inverse_bit_reversal_permutation(N):
    brp = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[brp] = np.arange(N)
    return inv


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=int).copy()
    n = len(x)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    brp = bit_reversal_permutation(n)
    return x[brp]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于测试）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    Gf = np.array([[1]], dtype=int)
    for _ in range(n):
        Gf = np.kron(Gf, F)
    brp = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(brp):
        B[i, j] = 1
    return (B @ Gf) % 2
