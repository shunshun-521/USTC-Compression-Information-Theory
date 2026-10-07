"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx >> np.arange(n)) & 1).sum(axis=1) if False else None
    rev = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for d in range(n):
        step = 1 << d
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u.astype(np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode u=", u, "-> x=", x)
