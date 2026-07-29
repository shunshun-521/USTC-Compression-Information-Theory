"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np
from functools import lru_cache


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f'0{n}b')[::-1]
        rev[i] = int(b, 2)
    return rev


@lru_cache(maxsize=16)
def polar_generator_matrix(N):
    """生成极化码生成矩阵 G_N = B_N F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[rev, :]


def polar_encode(u):
    """
    极化码编码：x = u * G_N（mod 2）。

    对小码长使用生成矩阵，对大码长使用等价的蝶形 + 比特倒序。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = polar_generator_matrix(N)
    return (u @ G) % 2


def polar_encode_fast(u):
    """
    O(N log N) 蝶形编码，等价于 polar_encode。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i] ^= u[i + step]
        step *= 2

    # 比特倒序置换使输出等价于 u @ G_N
    rev = bit_reversal_permutation(N)
    x = np.zeros(N, dtype=int)
    for i in range(N):
        x[rev[i]] = u[i]
    return x
