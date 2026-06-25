"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形 XOR 编码（不含比特倒序）。"""
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    for step in [2 ** i for i in range(n)]:
        for i in range(0, len(u), 2 * step):
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
    u = np.asarray(u, dtype=int)
    N = len(u)
    encoded = _butterfly_encode(u)
    br = bit_reversal_permutation(N)
    return encoded[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于验证）。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    Fn = F.copy()
    for _ in range(n - 1):
        Fn = np.kron(Fn, F) % 2
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ Fn) % 2
