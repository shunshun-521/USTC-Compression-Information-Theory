"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    result = np.zeros(N, dtype=int)
    for bit in range(n):
        result |= ((indices >> bit) & 1) << (n - 1 - bit)
    return result


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：u[i] ^= u[i + step]，共 log2(N) 层，
    最后对比特倒序置换后的结果输出码字。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for i in range(start, start + half):
                u[i] ^= u[i + half]
        block = half
    return u[bit_reversal_permutation(N)]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    # 与 G_N = B_N F^{⊗n} 一致的标准结果
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
