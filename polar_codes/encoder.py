"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：rev[i] = bit_reverse(i, log2 N)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）结构，等价于 x = u @ G_N (mod 2)，G_N 为 F 的 n 次 Kronecker 积。
    与 permuted SCD 译码器配套（信道 LLR 与码字逐位对齐）。
    """
    v = np.asarray(u, dtype=np.int8).copy()
    N = len(v)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        v_new = v.copy()
        for i in range(0, N, 2 * step):
            for j in range(step):
                v_new[i + j] = (v[i + j] + v[i + j + step]) & 1
                v_new[i + j + step] = v[i + j + step]
        v = v_new
        step *= 2
    return v
