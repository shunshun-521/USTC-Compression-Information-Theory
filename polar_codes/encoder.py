"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    return ((indices & 1) << (n - 1) |
            ((indices >> 1) & 1) << (n - 2) |
            ((indices >> 2) & 1) << (n - 3) |
            ((indices >> 3) & 1) << (n - 4) |
            ((indices >> 4) & 1) << (n - 5) |
            ((indices >> 5) & 1) << (n - 6) |
            ((indices >> 6) & 1) << (n - 7) |
            ((indices >> 7) & 1) << (n - 8) |
            ((indices >> 8) & 1) << (n - 9) |
            ((indices >> 9) & 1) << (n - 10))


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for step in range(n):
        s = 1 << step
        for i in range(0, N, 2 * s):
            u[i:i + s] ^= u[i + s:i + 2 * s]

    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    r"""构建生成矩阵 G_N = B_N F^{\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        Fn = np.kron(Fn, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=np.int8)[br]
    return (B @ Fn) % 2
