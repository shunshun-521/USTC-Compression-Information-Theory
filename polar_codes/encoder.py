"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    reversed_indices = ((indices & 1) << (n - 1))
    for bit in range(1, n):
        reversed_indices |= ((indices >> bit) & 1) << (n - 1 - bit)
    return reversed_indices


def polar_encode(u):
    """
    极化码编码（蝶形结构，输出等价于 u @ B_N @ F^⊗n）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    n = int(np.log2(N))

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                x[a] ^= x[b]
        step <<= 1

    return x


def polar_encode_with_br(u):
    """带比特倒序置换的编码（部分教材定义）。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
