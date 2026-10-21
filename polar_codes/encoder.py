"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode_butterfly(u):
    """蝶形编码 u * F^{\\otimes n}（不含比特倒序）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码。默认不含输出比特倒序，与 SC/SCL/BP 因子图一致；
    apply_bit_reversal=True 时 x = (u * F^{\\otimes n}) 再经 B_N 置换。
    """
    v = polar_encode_butterfly(u)
    if apply_bit_reversal:
        return v[bit_reversal_permutation(len(v))]
    return v


def build_codeword(info_bits, info_indices, N):
    """将 K 个信息比特填入 u 向量（冻结位为 0）"""
    u = np.zeros(N, dtype=int)
    u[info_indices] = info_bits
    return u
