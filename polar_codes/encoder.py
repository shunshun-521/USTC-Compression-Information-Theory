"""
极化码编码器
编码：x = u * F^{⊗n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def build_generator_matrix(N):
    """Arikan 生成矩阵 F^{⊗n}，F = [[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(F, G) % 2
    return G


def polar_encode(u):
    """
    极化码编码（蝶形，与 F^{⊗n} 矩阵乘法等价）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError(f"N={N} must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] = u[idx] ^ u[idx + half]
        block = half

    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵形式编码 x = u @ G (mod 2)"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    return (u @ build_generator_matrix(N)) % 2
