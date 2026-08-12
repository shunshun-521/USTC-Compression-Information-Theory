"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)],
        dtype=int,
    )


def polar_encode(u):
    """
    极化码编码（polarcodes 风格，无输出比特倒序）。
    使用 Arikan 核 F=[[1,1],[0,1]] 的蝶形结构。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N

    for _ in range(n):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def build_generator_matrix(N):
    """构建 GF(2) 生成矩阵 F^{\\otimes n}（无比特倒序）。"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G
