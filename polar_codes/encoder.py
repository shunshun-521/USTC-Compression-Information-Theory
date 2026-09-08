"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形：(u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    最后对比特倒序置换后的输出码字索引重排。
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return x[br]


def polar_encode_core(u):
    """蝶形编码（不含输出比特倒序），供 BP 早停重编码一致性校验。"""
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return x[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]
