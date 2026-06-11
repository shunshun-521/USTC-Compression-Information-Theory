"""
极化码编码器
编码：x = u * F^{⊗ n}（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码：x = u * F^{⊗ n}（mod 2）

    蝶形：每层上半支 XOR 下半支，共 log2(N) 层。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

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
