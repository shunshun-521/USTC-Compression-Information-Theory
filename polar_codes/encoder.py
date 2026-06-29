"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形递归结构，O(N log N)）。

    采用标准 Arikan 蝶形变换：对每一层将左半分区与右半分区做模二加，
    输出即为信道码字（比特倒序在 SC 译码器的译码顺序中处理）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block_size = N
    while block_size > 1:
        half = block_size // 2
        for base in range(0, N, block_size):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block_size = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    # 标准 Arikan 蝶形编码结果
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("编码器校验通过")
