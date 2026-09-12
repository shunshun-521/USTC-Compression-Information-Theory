"""
极化码编码器
编码：x = F^{\\otimes n} u，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 Permuted SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    n = N
    for _ in range(int(np.log2(N))):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u.astype(int)


def build_generator_matrix(N):
    """构建 Arikan 生成矩阵 F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    return G
