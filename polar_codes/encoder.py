"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，对应 x = u * F^{\\otimes n}）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2

    return u.astype(int)


if __name__ == "__main__":
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
