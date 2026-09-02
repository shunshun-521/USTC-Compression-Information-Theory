"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reverse_index(i, n):
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reverse_index(i, n) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N = B_N F^{\\otimes n}"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ G) % 2


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。
    将信息位与冻结位组成的源向量 u 编码为码字 x。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half
    return u


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print('polar_encode:', x)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print('matrix:', x_mat)
