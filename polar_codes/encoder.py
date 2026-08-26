"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def bit_reversed_index(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_transform_core(u):
    """蝶形极化变换。"""
    v = np.asarray(u, dtype=np.int8, copy=True)
    n = int(np.log2(len(v)))
    step = 1
    for _ in range(n):
        for i in range(0, len(v), 2 * step):
            left = v[i : i + step]
            right = v[i + step : i + 2 * step]
            v[i : i + step] = left ^ right
        step <<= 1
    return v


def polar_encode(u):
    """
    极化码编码。
    采用与译码器一致的蝶形变换（极化核 F^{⊗n}），
    译码端通过比特倒序调度实现等价的 B_N F^{⊗n} 结构。
    """
    return polar_transform_core(u)


def prepare_decoder_llr(llr_ch):
    """保留接口：当前译码器直接使用信道 LLR。"""
    return np.asarray(llr_ch, dtype=np.float64)
