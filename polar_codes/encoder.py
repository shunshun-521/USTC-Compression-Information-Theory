"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int32)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int32).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = u[i + j]
                b = u[i + j + step]
                u[i + j] = a ^ b
                u[i + j + step] = b

    brp = bit_reversal_permutation(N)
    return u[brp]


def polar_encode_no_brv(u):
    """不含比特倒序的编码（用于内部校验）"""
    u = np.asarray(u, dtype=np.int32).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = u[i + j]
                b = u[i + j + step]
                u[i + j] = a ^ b
                u[i + j + step] = b
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    assert np.array_equal(polar_encode(np.array([0, 0, 1, 1])), [0, 0, 1, 1])
    print("Encoder test passed.")
