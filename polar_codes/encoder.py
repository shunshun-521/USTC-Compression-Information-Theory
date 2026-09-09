"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8).copy()
    n = len(u)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("u length must be a power of 2")

    step = 1
    while step < n:
        for left in range(0, n, 2 * step):
            u[left:left + step] ^= u[left + step:left + 2 * step]
        step <<= 1
    return u


def polar_encode_with_bit_reversal(u):
    """含输出比特倒序置换的编码（x = u * G_N，G_N = B_N F^{\\otimes n}）"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于校验）"""
    f = np.array([[1, 0], [1, 1]], dtype=np.int8)
    g = f.copy()
    m = int(np.log2(N))
    for _ in range(m - 1):
        g = np.kron(g, f)

    br = bit_reversal_permutation(N)
    b = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        b[i, j] = 1
    return (b @ g) % 2
