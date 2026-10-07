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
        r = 0
        for b in range(n):
            r |= ((i >> b) & 1) << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（蝶形结构）。
    默认不做输出比特倒序，与因子图/SC 译码节点顺序一致；
    apply_bit_reversal=True 时在输出端施加 B_N。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1
    if apply_bit_reversal:
        br = bit_reversal_permutation(N)
        return u[br]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2, apply_bit_reversal=True)
    assert np.array_equal(x2, [1, 0, 1, 1]), f"带倒序编码错误: {x2}"
    print("encoder tests passed")
