"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形 XOR，与译码器一致的约定）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
        step *= 2
    return x.astype(int)


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i, j in enumerate(br):
        B[i, j] = 1
    G = (B @ F_n) % 2
    return (u @ G) % 2
