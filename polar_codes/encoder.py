"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for b in range(n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def polar_encode_core(u):
    """蝶形编码（不含比特倒序），与译码器内部比特顺序一致。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step //= 2
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = polar_encode_core(u)
    brp = bit_reversal_permutation(len(u))
    return u[brp]


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于测试）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    brp = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    B[np.arange(N), brp] = 1
    return (B @ G) % 2
