"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u, apply_bit_reversal=True):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否在蝶形运算后做比特倒序置换

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("Length must be a power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step <<= 1

    if apply_bit_reversal:
        return u[bit_reversal_permutation(N)]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x_br = polar_encode(u, apply_bit_reversal=True)
    x = polar_encode(u, apply_bit_reversal=False)
    print("u=", u, "x (no BR)=", x, "x (BR)=", x_br)
