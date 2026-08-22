"""
极化码编码器
编码：蝶形 XOR 结构，O(N log N) 复杂度（与 py-polar-codes 一致，不含比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，不含输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    stage_n = n
    while stage_n > 1:
        n_split = stage_n // 2
        for p in range(0, n, stage_n):
            for k in range(n_split):
                l = p + k
                u[l] = u[l] ^ u[l + n_split]
        stage_n = n_split
    return u


if __name__ == "__main__":
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "x:", x)
    assert np.array_equal(x, [0, 1, 0, 1]), f"编码器错误: {x}"

    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    print("u:", u2, "x:", x2)
    assert np.array_equal(x2, [1, 1, 0, 1]), f"编码器错误: {x2}"
