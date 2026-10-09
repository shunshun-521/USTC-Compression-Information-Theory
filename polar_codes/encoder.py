"""
极化码编码器
编码：分块 polar 变换（与标准因子图一致），O(N log N)
"""
import math

import numpy as np


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(math.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=np.int64)


def _polar_combine_block(a, b):
    """将两段长度为 m 的块合并为 polar 变换输出 [a xor b, b]。"""
    out = np.empty(len(a) + len(b), dtype=np.int8)
    out[: len(a)] = (a ^ b).astype(np.int8)
    out[len(a) :] = b.astype(np.int8)
    return out


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    nbits = int(math.log2(N))
    assert 2 ** nbits == N

    m = 1
    stages = nbits  # log2(N) 层
    for _ in range(stages):
        for i in range(0, N, 2 * m):
            a = u[i : i + m]
            b = u[i + m : i + 2 * m]
            u[i : i + 2 * m] = _polar_combine_block(a, b)
        m *= 2

    return u.astype(np.int8)


def polar_generator_matrix(N):
    """通过编码单位向量构造生成矩阵 G_N。"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[i] = 1
        G[i] = polar_encode(e)
    return G.T  # x = u @ G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("encode:", x)
    print("u@G:  ", x_ref)
