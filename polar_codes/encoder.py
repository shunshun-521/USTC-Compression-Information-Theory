"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bitrev(i)"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：Arikan 蝶形 XOR（与 Permuted SCD 配套，不在末尾做比特倒序）。
    等价于 x = u * F^{\\otimes n}（模 2）。
    """
    v = np.asarray(u, dtype=int).copy()
    N = len(v)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                v[p + k] ^= v[p + k + half]
        block = half
    return v


def polar_generator_matrix(N):
    """生成矩阵 F^{\\otimes n}（与 polar_encode 一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2
