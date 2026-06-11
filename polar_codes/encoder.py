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
    极化码编码（蝶形左支 XOR，与 SC 译码器比特倒序处理配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（信道传输顺序）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n_len = len(u)
    n = int(np.log2(n_len))
    if 2 ** n != n_len:
        raise ValueError(f"Length {n_len} must be a power of 2")

    block = n_len
    while block > 1:
        half = block // 2
        for start in range(0, n_len, block):
            for j in range(half):
                idx = start + j
                u[idx] ^= u[idx + half]
        block = half

    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
