"""
极化码编码器
编码：x = u * F^⊗n（蝶形），与标准 SC 因子图一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("N must be a power of 2")
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int
    )


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，分块阶段实现，O(N log N)）。
    与译码端比特倒序相位调度配套，等效于包含 B_N 的生成矩阵。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        if block == 1:
            break
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_with_br(u):
    """显式比特倒序置换后的码字（部分教材定义）。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]
