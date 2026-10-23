"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}，F = [[1,1],[0,1]]
"""
import numpy as np

_G_CACHE = {}


def bit_reversed(x, n):
    """单整数比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=np.int64)


def build_generator_matrix(N):
    """构建 Arikan 生成矩阵 F^{\\otimes n}。"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    f = np.array([[1, 1], [0, 1]], dtype=np.int8)
    g = f.copy()
    while g.shape[0] < N:
        g = np.kron(f, g) % 2
    _G_CACHE[N] = g
    return g


def polar_encode(u):
    """
    极化码编码（与 Arikan 蝶形等价，原地 XOR 实现）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    stage_len = n
    while stage_len > 1:
        n_split = stage_len // 2
        for p in range(0, n, stage_len):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        stage_len = n_split
    return u
