"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形：左支路 ^= 右支路）。
    译码器在比特倒序顺序下恢复 u，与信道符号一一对应。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    stage_n = N
    while stage_n > 1:
        n_split = stage_n // 2
        for p in range(0, N, stage_n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        stage_n = n_split
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder OK:", x)
