"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形编码（不含比特倒序）"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    N = len(u)
    x = _butterfly_encode(u)
    rev = bit_reversal_permutation(N)
    return x[rev]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    rev = bit_reversal_permutation(N)
    for i, r in enumerate(rev):
        B[i, r] = 1
    GN = (G @ B) % 2
    return (u @ GN) % 2


def channel_llr_to_decoder(llr_ch):
    """
    将信道 LLR（对应 polar_encode 输出顺序）转换为 SC/SCL 译码器所需顺序。
    """
    N = len(llr_ch)
    inv = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv]
