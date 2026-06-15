"""
极化码编码器
编码：x = u * F^{⊗ n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，等价于 x = u @ F^{⊗ n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            for j in range(step):
                u[start + j] ^= u[start + j + step]
        step <<= 1
    return u


def polar_encode_with_bit_reversal(u):
    """含输出比特倒序置换的编码（x = u @ B_N @ F^{⊗ n}）。"""
    return polar_encode(u)[bit_reversal_permutation(len(u))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
