"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[l] = u[l] XOR u[l + step]（Arikan 标准）
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.array(u, dtype=np.int8, copy=True)
    n = len(u)
    if n == 0 or (n & (n - 1)):
        raise ValueError("u length must be a power of 2")

    step = n
    while step > 1:
        half = step // 2
        for base in range(0, n, step):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        step = half

    br = bit_reversal_permutation(n)
    return u[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
