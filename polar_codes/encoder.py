"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev[:, ::-1]
    return (rev * (2 ** np.arange(n))).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（Kronecker 变换，非比特倒序约定，与译码器一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start:start + step] ^= x[start + step:start + 2 * step]
        step <<= 1
    return x.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
