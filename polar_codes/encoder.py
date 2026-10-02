"""
极化码编码器
编码：x = u * F^⊗n（蝶形 XOR，非比特倒序约定，与 SC/SCL 译码器一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（Kronecker F^{\otimes n}，O(N log N) 蝶形 XOR）。
    """
    x = np.asarray(u, dtype=np.uint8).copy() % 2
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step <<= 1
    return x.astype(int)


def polar_encode_generator_matrix(N):
    """生成矩阵 G_N（无比特倒序列置换）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    return G


def reorder_llr_for_decoder(llr_ch):
    """非比特倒序约定下无需重排；保留接口以兼容旧脚本。"""
    return np.asarray(llr_ch, dtype=np.float64)


def reorder_bits_from_decoder(u_hat):
    return np.asarray(u_hat, dtype=int)
