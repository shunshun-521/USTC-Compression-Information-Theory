"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev = 0
        for j in range(n):
            rev = (rev << 1) | ((i >> j) & 1)
        indices[i] = rev
    return indices


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for i in range(n):
        step = 1 << (n - i - 1)
        for j in range(0, N, 2 * step):
            for k in range(step):
                u[j + k] ^= u[j + k + step]

    br = bit_reversal_permutation(N)
    x = np.empty(N, dtype=np.int8)
    for i in range(N):
        x[i] = u[br[i]]
    return x


def polar_encode_core(u):
    """蝶形编码（不含比特倒序），与 SC 因子图自然序一致。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n = n_split
    return u


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    # G = B * F^2 编码（含比特倒序）
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    x_core = polar_encode_core(u)
    assert np.array_equal(x_core, [1, 1, 0, 1]), f"核心编码错误: {x_core}"
