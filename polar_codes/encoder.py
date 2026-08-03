"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        stride = 1 << stage
        for i in range(0, N, 2 * stride):
            for j in range(stride):
                u[i + j] ^= u[i + j + stride]
    return u


def construct_generator_matrix(N):
    """构造极化码生成矩阵 G_N = F^{\\otimes n}（用于验证）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    m = int(np.log2(N))
    for _ in range(m - 1):
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = construct_generator_matrix(4)
    expected = u @ G % 2
    print(f"u={u} -> x={x}, G*u={expected}")
    assert np.array_equal(x, expected), f"编码器与生成矩阵不一致: {x} vs {expected}"
    print("编码器校验通过")
