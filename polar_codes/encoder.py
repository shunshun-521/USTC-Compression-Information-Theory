"""
极化码编码器
编码：对 u 做蝶形 XOR（Arikan 变换），O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                x[idx] ^= x[idx + half]
        block = half
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    # N=4 蝶形编码结果（与 G_N @ u 在 B_N 置换下等价）
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
