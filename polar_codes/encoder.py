"""
极化码编码器：x = u F^{⊗n}（GF(2) 蝶形），与 SC/SCL 因子图一致。
"""
import numpy as np


def bit_reversal_permutation(N):
    """比特倒序置换索引（构造/报告用）"""
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


def polar_encode(u):
    """极化码编码（蝶形 XOR，O(N log N)）"""
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start:start + step] ^= x[start + step:start + 2 * step]
        step <<= 1
    return x.astype(np.int8)


def prepare_channel_llr(llr_natural, N):
    """与译码器一致：自然序信道 LLR（无额外置换）"""
    return np.asarray(llr_natural, dtype=np.float64)
