"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_G_MATRIX_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _get_generator_matrix(N):
    """标准极化码生成矩阵 G_N = F^{\\otimes n}"""
    if N not in _G_MATRIX_CACHE:
        F = np.array([[1, 0], [1, 1]], dtype=int)
        G = F
        n = int(np.log2(N))
        for _ in range(n - 1):
            G = np.kron(G, F)
        _G_MATRIX_CACHE[N] = G
    return _G_MATRIX_CACHE[N]


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                x[idx] ^= x[idx + half]
        block = half
    return x


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=int)
    G = _get_generator_matrix(len(u))
    return (u @ G) % 2
