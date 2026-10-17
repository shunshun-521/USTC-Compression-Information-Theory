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

    蝶形结构（由粗到细划分，与 SC 因子图一致）：
        对块长 n 的子块，左半与右半异或；n 依次减半，共 log2(N) 层。
    最后对比特倒序置换（B_N）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        n = half
    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def build_generator_matrix(N):
    """构建生成矩阵（行 i 为单比特消息 polar_encode(e_i)）"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[i] = polar_encode(e)
    return G
