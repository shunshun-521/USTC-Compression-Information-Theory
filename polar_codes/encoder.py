"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{⊗n}，F = [[1,1],[0,1]]
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，无输出比特倒序）。
    信道发送的码字 x 与内部 u 向量同序；译码器在比特倒序索引上工作。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_generator_matrix(N):
    """生成极化码生成矩阵 G_N = F^{⊗n}，F=[[1,1],[0,1]]。"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        F_n = np.kron(F, F_n)
    return F_n
