"""
极化码编码器
编码：c = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        c: 长度为 N 的码字（c = u @ G_N mod 2）

    实现：蝶形结构，u[i] ^= u[i + step]
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    step = N
    while step > 1:
        step //= 2
        for p in range(0, N, 2 * step):
            for k in range(step):
                u[p + k] ^= u[p + k + step]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
