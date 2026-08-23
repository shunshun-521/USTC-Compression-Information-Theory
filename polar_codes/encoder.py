"""
极化码编码器
编码：x = F^{⊗n} u，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，O(N log N)）。

    每层对块内左半部分与右半部分做 XOR：u[l] ^= u[l + block/2]。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N

    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half

    return u.astype(int)


def build_generator_matrix(N):
    """构建生成矩阵 F^{⊗n}（Arikan 核 [[1,1],[0,1]]）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G
