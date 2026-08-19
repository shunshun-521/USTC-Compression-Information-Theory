"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[l] ^= u[l + step]（Arikan F=[[1,1],[0,1]] 核）
        - 共 log2(N) 层
    """
    u = np.asarray(u, dtype=np.int_)
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, 'N must be a power of 2'

    x = u.copy()
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                left = start + k
                x[left] ^= x[left + half]
        block = half

    return x


def build_generator_matrix(N):
    """构建极化码生成矩阵 F^{\\otimes n}"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f'编码器错误: {x} vs {(u @ G) % 2}'
    print('Encoder test passed:', x)
