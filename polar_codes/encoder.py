"""
极化码编码器
编码：x = u * A_N（蝶形），信道输出顺序与译码器一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回比特倒序置换索引：out[i] = u[bit_reverse(i)]"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：蝶形结构 O(N log N)，不做输出比特倒序
    （与 SC/SCL 译码器的自然信道索引一致）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
