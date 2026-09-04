"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（蝶形结构，无输出比特倒序）。

    每层对相邻块执行 (x XOR y, y) 合并。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N)) + 1
    m = 1
    for _ in range(n - 1):
        for i in range(0, N, 2 * m):
            x = u[i:i + m]
            y = u[i + m:i + 2 * m]
            u[i:i + 2 * m] = np.concatenate([(x ^ y) % 2, y % 2])
        m *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
