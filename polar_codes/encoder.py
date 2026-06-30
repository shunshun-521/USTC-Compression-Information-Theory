"""
极化码编码器
编码：x = u * F_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码：x = u * G_N，G_N = B_N F^{⊗n}。

    蝶形结构（O(N log N)）：
      每层相邻对 (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    比特倒序置换 B_N 由 SC/SCL 译码器在比特序上等价吸收，
    故编码端直接输出 u * F^{⊗n}（与 validate 中 u=[1,0,1,1]->[1,1,0,1] 一致）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u
