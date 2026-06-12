"""
极化码编码器
编码：x = u * F^{⊗n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u * F^{⊗n}（mod 2）。

    蝶形结构：上半分支 XOR 下半分支（与 SC 译码器配套）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            for j in range(start, start + step):
                u[j] ^= u[j + step]
        step *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
