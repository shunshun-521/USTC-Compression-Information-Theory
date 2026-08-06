"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode_butterfly(u):
    """蝶形编码（不含比特倒序）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = N
    while step > 1:
        half = step // 2
        for base in range(0, N, step):
            u[base:base + half] ^= u[base + half:base + step]
        step = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u_enc = polar_encode_butterfly(u)
    brp = bit_reversal_permutation(len(u_enc))
    return u_enc[brp]


def channel_llr_to_decoder(llr_ch):
    """
    将信道顺序 LLR 置换为译码器内部顺序（与蝶形编码域对齐）。
    """
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[brp]
