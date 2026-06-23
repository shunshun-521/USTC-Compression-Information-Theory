"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}（自然序，无比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    与生成矩阵 G_N = F^{\\otimes n} 一致：x = u @ G_N (mod 2)。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
    return x.astype(np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
