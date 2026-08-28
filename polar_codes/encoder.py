"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if n == 0:
        return np.array([0], dtype=int)
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


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
    n = int(np.log2(len(u)))
    if 2 ** n != len(u):
        raise ValueError("Length of u must be a power of 2")

    step = 1
    while step < len(u):
        for i in range(0, len(u), 2 * step):
            left = u[i:i + step]
            right = u[i + step:i + 2 * step]
            u[i:i + step] = left ^ right
        step *= 2

    br = bit_reversal_permutation(len(u))
    return u[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F2 = np.kron(F, F)
    G = F2[bit_reversal_permutation(4), :] % 2
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("Encoder test passed.")
