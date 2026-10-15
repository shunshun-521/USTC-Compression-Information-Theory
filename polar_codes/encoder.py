"""
极化码编码器
编码：x = u * F^{\otimes n}，蝶形结构 O(N log N)
（与串行抵消译码树的比特顺序一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan F^{\otimes n} 蝶形，无额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    for layer in range(1, n + 1):
        block = 1 << layer
        half = block // 2
        for blk in range(N // block):
            start = blk * block
            u[start : start + half] ^= u[start + half : start + block]
    return u


def polar_generator_matrix(N):
    """生成 F^{\otimes n}（用于校验）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)

    def f_power(n):
        if n == 0:
            return np.array([[1]], dtype=int)
        return np.kron(f_power(n - 1), F) % 2

    n = int(np.log2(N))
    return f_power(n)
