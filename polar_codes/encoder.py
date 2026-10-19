"""
极化码编码器
编码：x = u * F_N（Kronecker 积），蝶形 XOR，O(N log N)
与 SC/SCL 译码器采用相同的非比特倒序约定（与 3GPP/CommPy 一致）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（Kronecker F^{⊗n} 变换，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位应为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            u[start: start + step] ^= u[start + step: start + 2 * step]
        step <<= 1
    return u.astype(np.int8)


def polar_encode_with_bitrev(u):
    """编码后再做比特倒序置换（仅在与信道索引匹配时使用）。。"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]
