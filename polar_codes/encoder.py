"""
极化码编码器
编码：蝶形结构 O(N log N)，与标准极化变换一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f'{i:0{n}b}'[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    v = np.asarray(u, dtype=np.int8).copy()
    N = len(v)
    n = int(np.log2(N))
    for layer in range(1, n + 1):
        block = 1 << layer
        half = block // 2
        for blk in range(N // block):
            start = blk * block
            v[start:start + half] ^= v[start + half:start + block]
    return v
