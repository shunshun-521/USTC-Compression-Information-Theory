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
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("Length of u must be a power of 2")

    for stage in range(n):
        step = 1 << (n - stage - 1)
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]

    rev = bit_reversal_permutation(N)
    return u[rev]


def prepare_channel_llr(llr_ch):
    """
    将信道 LLR 按比特倒序重排，以匹配编码器输出顺序与译码树。
    """
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]
