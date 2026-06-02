"""
极化码编码器
编码：x = u * F^⊗n（Arikan 核），蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def build_generator_matrix(N):
    """构造 Arikan 生成矩阵 F^⊗n，F = [[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(F, G)
    return G


def polar_encode(u):
    """
    极化码编码（Arikan 核蝶形，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        if block == 1:
            break
        half = block // 2
        for start in range(0, N, block):
            for j in range(half):
                u[start + j] ^= u[start + j + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u, "x:", x, "u@G:", x_ref)
    assert np.array_equal(x, x_ref)
