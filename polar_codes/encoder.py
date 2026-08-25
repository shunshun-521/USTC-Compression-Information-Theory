"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，不做输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]

    return u


def prepare_decoder_llr(llr_ch):
    """
    信道 LLR 直接对应编码输出顺序，无需额外置换。
    保留此函数以兼容仿真脚本接口。
    """
    return np.asarray(llr_ch, dtype=np.float64)
