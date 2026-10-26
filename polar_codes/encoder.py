"""
极化码编码器（与 Permuted SCD 配套）
非递归 XOR 左分区编码，等价于 Arikan 生成矩阵作用于 u。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("N must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码：按阶段将右半分区 XOR 到左半分区（mod 2）。
    apply_bit_reversal 保留接口；Permuted SCD 下默认不对码字再置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        block = half
    if apply_bit_reversal:
        br = bit_reversal_permutation(N)
        u = u[br]
    return u.astype(int)


def build_codeword(u_payload, info_indices, N):
    u = np.zeros(N, dtype=int)
    u[info_indices] = u_payload
    return polar_encode(u)
