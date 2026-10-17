"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗n}，蝶形实现 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx & 1) << (n - 1))
    for i in range(1, n):
        rev |= ((idx >> i) & 1) << (n - 1 - i)
    return rev


def polar_encode(u):
    """
    极化码编码：x = u * F^{⊗n}（与标准生成矩阵一致，无额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    stage = N
    while stage > 1:
        half = stage // 2
        for base in range(0, N, stage):
            u[base : base + half] ^= u[base + half : base + stage]
        stage = half
    return u.astype(int)
