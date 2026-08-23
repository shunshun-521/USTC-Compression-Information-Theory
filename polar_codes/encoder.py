"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构 + 比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Length {n} must be a power of 2")

    layers = int(np.log2(n))
    for layer in range(layers):
        step = 1 << layer
        for i in range(0, n, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    br = bit_reversal_permutation(n)
    return u[br].astype(int)


def polar_encode_no_br(u):
    """蝶形编码，不做输出比特倒序（与部分 SC 实现配套）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    layers = int(np.log2(n))
    for layer in range(layers):
        step = 1 << layer
        for i in range(0, n, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    x2 = polar_encode_no_br(u)
    print("polar_encode_no_br:", x2)
