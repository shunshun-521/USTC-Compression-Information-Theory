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
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def bit_reversed_index(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Kronecker 蝶形，与 SC 译码器配套）。

    按块大小 N, N/2, ..., 2 依次将右半区 XOR 到左半区。
    """
    v = np.asarray(u, dtype=np.int8).copy()
    n = len(v)
    if n & (n - 1):
        raise ValueError("码长 N 必须为 2 的幂")
    block = n
    while block > 1:
        half = block // 2
        for p in range(0, n, block):
            for k in range(half):
                idx = p + k
                v[idx] ^= v[idx + half]
        block = half
    return (v % 2).astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    # 与 Arikan 核 [[1,1],[0,1]] 的 Kronecker 编码一致
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder test passed")
