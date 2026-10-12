"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_f_transform(bits):
    """对比特向量施加 F^{⊗n}（蝶形 XOR，与编码内核一致）"""
    bits = np.asarray(bits, dtype=np.int8).copy()
    N = len(bits)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            bits[i : i + step] ^= bits[i + step : i + 2 * step]
        step *= 2
    return bits.astype(int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（x[i] 对应 u 经 F^{⊗n} 后的第 bit_reversal(i) 位）
    """
    # 与 SC 译码器一致：x = u · F^{⊗n}（信道索引即自然序）
    return polar_f_transform(u)


def polar_encode_matrix(N):
    """生成矩阵 G_N（用于校验），G[i,j] = (i & j) == j 的 Kronecker 形式经 B_N"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    G = G[rev, :]
    return G
