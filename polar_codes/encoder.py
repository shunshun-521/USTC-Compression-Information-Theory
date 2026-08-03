"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def get_generator_matrix(N):
    """获取极化码生成矩阵 G_N = F^{\\otimes n}"""
    if N not in _G_CACHE:
        n = int(np.log2(N))
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        G = np.array([[1]], dtype=np.int8)
        for _ in range(n):
            G = np.kron(G, F) % 2
        _G_CACHE[N] = G
    return _G_CACHE[N]


def polar_encode(u):
    """
    极化码编码（蝶形结构 O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    length = x.size
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start:start + step] ^= x[start + step:start + 2 * step]
        step *= 2
    return x % 2
