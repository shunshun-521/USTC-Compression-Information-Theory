"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in indices:
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed(i, n):
    """将 i 的 n 位二进制表示做比特倒序"""
    result = 0
    for k in range(n):
        if (i >> k) & 1:
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 译码器配套，输出不做比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1

    return u


if __name__ == "__main__":
    # 标准蝶形编码: u=[0,1,0,1] -> x=[0,0,1,1]
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
