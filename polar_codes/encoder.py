"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int64)
    rev = rev[:, ::-1]
    weights = 1 << np.arange(n)
    return (rev * weights).sum(axis=1)


def polar_transform(u):
    """Kronecker F^{⊗n} 蝶形变换（不含比特倒序）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1
    return u.astype(int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换 B_N）。
    x = B_N * F^{⊗n} * u
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    x = polar_transform(u)
    br = bit_reversal_permutation(N)
    return x[br].astype(int)


def channel_llr_to_decoder(llr_ch):
    """将信道顺序的 LLR 重排为 SC/SCL/BP 译码器输入顺序"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]
