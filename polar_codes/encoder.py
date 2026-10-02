"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reverse_index(i, n):
    r = 0
    for b in range(n):
        if (i >> b) & 1:
            r |= 1 << (n - 1 - b)
    return r


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reverse_index(i, n) for i in range(N)], dtype=int)


def inverse_bit_reversal_permutation(N):
    """比特倒序的逆置换索引"""
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[br] = np.arange(N)
    return inv


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br]


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=np.int8)[br]
    return (B @ G) % 2
