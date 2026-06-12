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
    极化码编码（pipelined Arikan 结构，x = u @ G_N）。

    采用 even/odd 交织蝶形，与标准极化码生成矩阵 G_N = F^{\\otimes n} 一致。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    output = u.copy()
    tmp = np.zeros(N, dtype=int)
    half = N // 2
    i_even = np.arange(0, N, 2)
    i_odd = np.arange(1, N, 2)

    for _ in range(n):
        tmp[:half] = output[i_even]
        tmp[half:] = output[i_odd]
        output[i_odd] = tmp[i_odd]
        output[i_even] = tmp[i_even] ^ tmp[i_odd]
    return output


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
