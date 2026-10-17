"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（与标准蝶形 XOR 顺序一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否在编码后对码字做比特倒序置换

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split

    if apply_bit_reversal:
        br = bit_reversal_permutation(N)
        x = u[br]
    else:
        x = u
    return x.astype(int)


def polar_generator_matrix(N, apply_bit_reversal=False):
    """生成 G_N（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    if apply_bit_reversal:
        br = bit_reversal_permutation(N)
        G = G[br, :]
    return G % 2
