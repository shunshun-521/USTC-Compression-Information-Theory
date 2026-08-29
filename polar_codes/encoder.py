"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def bit_reversed(x, n):
    """对标量索引 x 做比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    采用逐级分块 XOR 蝶形运算，与 SC 译码器（按比特倒序处理）配套使用。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "-> x:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
