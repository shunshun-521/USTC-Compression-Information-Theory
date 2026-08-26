"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 py-polar-codes 一致）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = N
    while step > 1:
        step //= 2
        for p in range(0, N, step * 2):
            for k in range(step):
                u[p + k] ^= u[p + k + step]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 0, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    assert np.array_equal(x, [0, 1, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
