"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    step = 1
    for _ in range(n):
        for i in range(0, len(u), 2 * step):
            for j in range(i, i + step):
                u[j] = u[j] ^ u[j + step]
        step *= 2

    return u[bit_reversal_permutation(len(u))]


def polar_generator_matrix(N):
    """生成矩阵 G_N = B_N F^{\\otimes n}，用于验证。"""
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=int)
    f_n = f.copy()
    for _ in range(n - 1):
        f_n = np.kron(f_n, f)
    return f_n[bit_reversal_permutation(N)]
