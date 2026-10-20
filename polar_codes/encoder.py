"""
极化码编码器
编码：对 u 做蝶形 XOR（等价于 x = u @ F^{\\otimes n}），复杂度 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r = (r << 1) | ((i >> b) & 1)
        rev[i] = r
    return rev


def bit_reversed(i, n):
    """单索引比特倒序（与 polarcodes 一致）"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码：原地蝶形 XOR，输出码字 x（长度 N）。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 蝶形编码 u=[1,0,1,1] -> x=[1,1,0,1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder OK:", x)
