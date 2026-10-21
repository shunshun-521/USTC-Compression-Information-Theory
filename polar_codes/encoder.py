"""
极化码编码器
编码：x = u * F^{\\otimes n}（蝶形结构，O(N log N)）
与 SC/SCL 译码器采用相同的非比特倒序约定（与 G_N = B_N F^{\\otimes n} 中 B_N 在译码端对齐）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1
    return u.astype(int)


def polar_encode_with_bit_reversal(u):
    """蝶形编码后再做比特倒序置换（仅用于与 G=B_N F^{\\otimes n} 矩阵对照）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


def generator_matrix(N):
    """返回 G_N = B_N F^{\\otimes n}（用于校验）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i in range(N):
        B[i, br[i]] = 1
    return (B @ G) % 2
