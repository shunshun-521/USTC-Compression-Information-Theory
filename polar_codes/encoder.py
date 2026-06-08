"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    reversed_indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev = 0
        val = i
        for _ in range(n):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        reversed_indices[i] = rev
    return reversed_indices


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i:i + step] = u[i:i + step] ^ u[i + step:i + 2 * step]

    br = bit_reversal_permutation(N)
    return u[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("编码器校验通过")
