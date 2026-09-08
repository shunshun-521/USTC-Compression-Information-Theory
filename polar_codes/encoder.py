"""
极化码编码器
编码：蝶形 XOR 结构，复杂度 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：x_out[i] = x_in[indices[i]]"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        bits = format(i, f"0{n}b")[::-1]
        rev[i] = int(bits, 2)
    return rev


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序。"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与标准 Arıkan 生成矩阵一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_with_br(u):
    """带输出比特倒序的编码（部分文献约定）。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]
