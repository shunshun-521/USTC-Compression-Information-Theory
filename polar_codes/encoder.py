"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reversal_indices(N):
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n)) & 1
    rev_bits = bits[:, ::-1]
    powers = 1 << np.arange(n)
    return (rev_bits * powers).sum(axis=1)


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_reversal_indices(N)


def polar_encode(u):
    """
    极化码编码（Arikan 非递归块蝶形，与 SC/SCL 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode:", polar_encode(u))
