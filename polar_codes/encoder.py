"""
极化码编码器
编码：x = u * G_N，蝶形结构 O(N log N)
与 Permuted SC 译码器配套（不在末尾做比特倒序；等价于标准 B_N F^{\\otimes n} 的一种实现约定）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int
    )


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 mcba1n / 5G 常用实现一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_block = N
    while n_block > 1:
        half = n_block // 2
        for p in range(0, N, n_block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        n_block = half
    return u.astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}，F=[[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(F, G)
    return G % 2


def encode_self_test():
    """N=4 编码与生成矩阵一致"""
    u = np.array([1, 0, 1, 1])
    G = polar_generator_matrix(4)
    x = polar_encode(u)
    return np.array_equal(x, (u @ G) % 2)
