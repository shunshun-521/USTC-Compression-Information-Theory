"""
极化码编码器
编码：u 经蝶形结构得到码字 x（与标准 polar_encode 一致，无额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(int)
    rev = rev[:, ::-1]
    powers = 2 ** np.arange(n)
    return (rev * powers).sum(axis=1)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（O(N log N) 蝶形结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_matrix(u):
    """基于生成矩阵 G = B_N F^{\\otimes n} 的编码（用于校验）。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2
