"""
极化码编码器
编码：利用 Arikan 核 F=[[1,1],[0,1]] 的蝶形结构，复杂度 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = bit_reversed(i, n)
    return rev


def bit_reversed(x, n):
    """对标量索引做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（非系统化，与 SC 译码器配套）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    # 与 Arikan 核 F=[[1,1],[0,1]] 的矩阵乘法一致
    F = np.array([[1, 1], [0, 1]], dtype=int)
    F2 = np.kron(F, F) % 2
    assert np.array_equal(x, u @ F2 % 2), f"编码器错误: {x}"
    print("Encoder test passed.")
