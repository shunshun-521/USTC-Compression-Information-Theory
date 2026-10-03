r"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    等价于 u @ G_N (mod 2)，G_N = B_N F^⊗n。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    n = len(x)
    if n & (n - 1):
        raise ValueError("length must be power of 2")
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
        step <<= 1
    perm = bit_reversal_permutation(n)
    return x[perm].astype(np.int8)


def build_generator_matrix(N):
    """G_N = B_N F^⊗n，用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = np.array([[1]], dtype=np.int8)
    while Fn.shape[0] < N:
        Fn = np.kron(Fn, F)
    perm = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, p in enumerate(perm):
        B[i, p] = 1
    return (B @ Fn) % 2
