"""
极化码编码器
编码：x = u * F_N（蝶形/light_encode），与标准极化码生成矩阵一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r |= ((i >> b) & 1) << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。
    与 aff3ct Encoder_polar::light_encode 等价。
    """
    bits = np.asarray(u, dtype=np.int8).copy()
    N = len(bits)
    k = N >> 1
    while k > 0:
        for j in range(0, N, 2 * k):
            for i in range(k):
                bits[j + i] ^= bits[k + j + i]
        k >>= 1
    return bits


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    assert np.array_equal(x, [1, 1, 1, 1]), f"编码器错误: {x}"
