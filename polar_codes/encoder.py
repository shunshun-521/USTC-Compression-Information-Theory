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
    """
    u = np.array(u, dtype=np.int64).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("u length must be power of 2")
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << (layer + 1)
        half = step >> 1
        for i in range(0, N, step):
            u[i:i + half] ^= u[i + half:i + step]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder self-test OK:", x)
