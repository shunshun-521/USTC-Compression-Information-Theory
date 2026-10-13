"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int64)
    rev = rev[:, ::-1]
    powers = 2 ** np.arange(n)
    return (rev * powers).sum(axis=1)


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（蝶形 XOR，可选 B_N 比特倒序置换）。

    默认不施加 B_N，与 SC/SCL 译码器及 G=F^{\\otimes n} 一致。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("Length of u must be a power of 2")

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]

    if apply_bit_reversal:
        br = bit_reversal_permutation(N)
        x = u[br]
    else:
        x = u
    return x.astype(np.int8)


def polar_generator_matrix(N):
    """GF(2) 生成矩阵 G_N = F^{\\otimes n}（与默认编码一致）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G
