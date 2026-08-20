"""
极化码编码器
编码：x = u * G_N，G_N = B_N * F^{\otimes n}
蝶形结构 O(N log N) + 比特倒序置换
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for j in range(n):
            r = (r << 1) | ((i >> j) & 1)
        rev[i] = r
    return rev


def _butterfly_transform(u):
    """u * F^{\otimes n} 蝶形运算"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 2 ** (stage + 1)
        half = step // 2
        for i in range(0, N, step):
            for j in range(i, i + half):
                u[j] ^= u[j + half]
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = (u * F^{\otimes n}) 经比特倒序行置换
    """
    u = _butterfly_transform(u)
    rev = bit_reversal_permutation(len(u))
    return u[rev]


def build_generator_matrix(N):
    """构建 G_N = B_N * F^{\otimes n}"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    rev = bit_reversal_permutation(N)
    return F_n[rev, :]
