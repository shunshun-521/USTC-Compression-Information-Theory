"""
极化码编码器
编码利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reverse_index(i, n):
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reverse_index(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    对相邻块 (x, y) 执行 [x XOR y, y] 合并，共 log2(N) 层。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    m = 1
    while m < N:
        for i in range(0, N, 2 * m):
            x = u[i : i + m]
            y = u[i + m : i + 2 * m]
            u[i : i + m] = x ^ y
        m *= 2
    return u


def compute_channel_permutation(N):
    """信道 LLR 与 SC 译码树顺序一致（恒等映射）"""
    return np.arange(N, dtype=int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
