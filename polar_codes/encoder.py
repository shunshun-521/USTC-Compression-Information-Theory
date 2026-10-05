"""
极化码编码器
编码：x = u * G_N，G_N = F^{\otimes n}（自然序，无额外比特倒序）
利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 G_N = F^{\\otimes n} 一致）。

    每层对块内所有 j：u[j] ^= u[j + step]
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    m = int(np.log2(N))
    for _ in range(m - 1):
        G = np.kron(G, F) % 2
    return G
