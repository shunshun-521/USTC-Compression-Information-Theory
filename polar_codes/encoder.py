"""
极化码编码器
编码：u -> v = u * F^{\\otimes n}（蝶形 XOR，大分组优先）
与 SC 译码器配套；不在此处做比特倒序（信道索引与 u 一致）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序（MSB 倒序，与常见极化码文献一致）。"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（与 u 同索引顺序）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N == 0 or (N & (N - 1)):
        raise ValueError("u length must be a positive power of 2")

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + half + k]
        block = half

    return u.astype(int)


def polar_encode_with_bit_reversal(u):
    """带输出比特倒序的编码（仅用于与 B_N 显式矩阵对照）。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def build_generator_matrix(N):
    """构造 F^{\\otimes n}（GF(2)），用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    m = int(np.log2(N)) - 1
    for _ in range(m):
        G = np.kron(G, F) % 2
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码（参考）。"""
    u = np.asarray(u, dtype=np.int8)
    G = build_generator_matrix(len(u))
    return (u @ G) % 2
