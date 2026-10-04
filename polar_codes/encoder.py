"""
极化码编码器
编码：蝶形结构 O(N log N)，与 Arikan 核 F=[[1,1],[0,1]] 的 Kronecker 积一致（无额外输出倒序）
"""
import numpy as np


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
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
    极化码编码（蝶形，分阶段合并子块）。
    """
    u = np.array(u, dtype=np.int8).copy()
    n_block = u.shape[0]
    n = n_block
    while n > 1:
        n_half = n // 2
        for start in range(0, n_block, n):
            for k in range(n_half):
                idx = start + k
                u[idx] = (u[idx] ^ u[idx + n_half]) & 1
        n = n_half
    return u


def arikan_generator(N):
    """生成矩阵 F^{⊗ n}，F=[[1,1],[0,1]]"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(F, G)
    return G % 2
