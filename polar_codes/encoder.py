"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度

实现采用 Arikan 标准非递归 XOR（块长 N, N/2, ...），
与 SC/SCL 因子图及 LLR 顺序一致。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx & 1) << (n - 1))
    for b in range(1, n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（信道传输顺序）
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


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[rev, :]
