"""
极化码编码器
编码：x = u * G_N（非比特倒序约定），蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Kronecker F^{⊗n}，与 SC/SCL 译码器索引一致）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    length = x.size
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
