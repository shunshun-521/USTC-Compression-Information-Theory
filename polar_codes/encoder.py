"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversed(x, n):
    """对索引 x 做 n 位比特倒序。"""
    if np.isscalar(x):
        result = 0
        for i in range(n):
            if x & (1 << i):
                result |= 1 << (n - 1 - i)
        return result
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros_like(x)
    for idx, val in np.ndenumerate(x):
        out[idx] = bit_reversed(int(val), n)
    return out


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=np.int64)


def _build_generator_matrix(N):
    """G_N = F^{\\otimes n}（与分层 XOR 编码一致）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码（分层 XOR，与标准 G_N 一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    for _ in range(N):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def polar_encode_matrix(u):
    """矩阵乘法编码 x = u @ G_N mod 2（校验用）。"""
    u = np.asarray(u, dtype=np.int8)
    G = _build_generator_matrix(len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("butterfly:", x, "matrix:", xm)
