"""
极化码编码器
编码：x = u * F^{\\otimes n}，蝶形结构 O(N log N)
（自然序因子图约定，输出不做比特倒序置换）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，自然信道序输出）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"Length {N} must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                x[idx] ^= x[idx + half]
        block = half

    return x


def polar_generator_matrix(N):
    """构造 F^{\\otimes n} 生成矩阵（用于验证）。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G
