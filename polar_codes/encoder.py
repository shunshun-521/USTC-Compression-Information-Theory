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
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构）。

    每层将向量分为两半 (a, b)，更新为 (a XOR b, b)。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    m = 1
    for _ in range(n):
        for i in range(0, N, 2 * m):
            a = u[i:i + m]
            b = u[i + m:i + 2 * m]
            u[i:i + 2 * m] = np.concatenate([(a ^ b) % 2, b])
        m *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    print("Encoder test: run decoder_sc validation for round-trip check.")
