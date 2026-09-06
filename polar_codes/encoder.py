r"""
极化码编码器
编码：x = G_N @ u，G_N = F^{\otimes n}（F = [[1,1],[0,1]]）
利用蝶形结构实现 O(N log N) 复杂度；比特倒序在译码阶段处理
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    block = n
    n_stages = int(np.log2(n))

    for _ in range(n_stages):
        if block == 1:
            break
        half = block // 2
        for start in range(0, n, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_generator_matrix(N):
    r"""构造极化码生成矩阵 G_N = F^{\otimes n}，F = [[1,1],[0,1]]"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (G @ u) % 2
    print("u =", u)
    print("polar_encode(u) =", x)
    print("G @ u =", x_ref)
    assert np.array_equal(x, x_ref)
