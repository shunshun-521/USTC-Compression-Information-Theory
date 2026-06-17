"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    采用与 SC 译码器匹配的蝶形 XOR 结构（按块减半迭代），
    输出即为信道传输码字 c（与 u 同索引顺序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    if N & (N - 1):
        raise ValueError(f"N={N} must be a power of 2")

    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        n = half

    return u.astype(np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
