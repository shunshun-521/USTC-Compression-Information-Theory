"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形递归结构，O(N log N)）。

    对每一层：将相邻块 (a, b) 映射为 (a XOR b, b)。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    m = 1
    for _ in range(n):
        for i in range(0, N, 2 * m):
            a = u[i:i + m]
            b = u[i + m:i + 2 * m]
            u[i:i + m] = a ^ b
            u[i + m:i + 2 * m] = b
        m *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"
    print("编码器校验通过")
