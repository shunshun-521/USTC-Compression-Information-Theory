"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（分块蝶形结构，与标准 SC/SCL/BP 译码器兼容）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for p in range(0, N, n):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        n = half
    return u


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N（由编码器导出，用于验证）"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        u = np.zeros(N, dtype=np.int8)
        u[i] = 1
        G[i] = polar_encode(u)
    return G
