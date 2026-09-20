"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，G_N = F^{⊗n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    n = len(x)
    assert n > 0 and (n & (n - 1)) == 0, "N must be a power of 2"

    block = n
    while block > 1:
        half = block // 2
        for start in range(0, n, block):
            for i in range(start, start + half):
                x[i] ^= x[i + half]
        block = half

    return x


if __name__ == "__main__":
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    print("u=", u, "-> x=", x)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
