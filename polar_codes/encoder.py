"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(bin(i)[2:].zfill(n)[::-1], 2)
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    return int(bin(i)[2:].zfill(n)[::-1], 2)


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 G_N = F^{\otimes n} 矩阵乘法等价）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N

    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_with_br(u):
    """编码后做比特倒序置换（部分文献约定）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    print(f"u={u} -> x={x}")
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
