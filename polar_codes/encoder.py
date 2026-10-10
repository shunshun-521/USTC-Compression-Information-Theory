"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx >> np.arange(n)) & 1).sum(axis=1) if False else None
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def _build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int64)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    brp = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int64)
    for i, j in enumerate(brp):
        B[i, j] = 1
    return (B @ G) % 2


def polar_encode_butterfly(u):
    """蝶形编码（不含比特倒序），u 会被原地修改"""
    u = np.array(u, dtype=np.int64, copy=True)
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    x = polar_encode_butterfly(u)
    brp = bit_reversal_permutation(N)
    return x[brp]


def polar_encode_matrix(u):
    """矩阵乘法编码 x = u G_N mod 2"""
    N = len(u)
    G = _build_generator_matrix(N)
    return (np.asarray(u, dtype=np.int64) @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("butterfly+BRP:", x)
    xm = polar_encode_matrix(u)
    print("matrix:", xm)
