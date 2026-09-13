"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    v = np.asarray(u, dtype=np.int8).copy()
    N = len(v)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            left = start
            right = start + half
            v[left:right] ^= v[right:right + half]
        block = half

    return v


def build_generator_matrix(N):
    """构造极化码生成矩阵 G_N = F^{\\otimes n}，F=[[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    G = build_generator_matrix(4)
    print("G @ u =", (G @ u) % 2)
