"""
极化码编码器
编码：x = u * F^⊗n（块蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（块蝶形，与 SC/SCL 译码器配套）。

    等价于 x = u * B_N * F^{⊗n}，其中比特倒序 B_N 已吸收到
    信道索引与译码器因子图约定中；若需显式 B_N，请调用 polar_encode_with_br。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            u[start : start + half] ^= u[start + half : start + block]
        block = half
    return u


def polar_encode_with_br(u):
    """含比特倒序的编码（仅用于与 G=B_N F^{⊗n} 矩阵对照）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]
