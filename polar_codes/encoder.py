"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N = F^{\\otimes n}"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F, F_n)
    return F_n


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    蝶形结构：u[l] ^= u[l + step]（上支更新），共 log2(N) 层。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    stage_len = N
    while stage_len > 1:
        half = stage_len // 2
        for p in range(0, N, stage_len):
            for k in range(half):
                l = p + k
                u[l] = (u[l] + u[l + half]) % 2
        stage_len = half
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=int)
    G = build_generator_matrix(len(u))
    return (u @ G) % 2
