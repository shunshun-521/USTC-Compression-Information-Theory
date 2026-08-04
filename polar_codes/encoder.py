"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_rev_indices(N):
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_rev_indices(N)


def _butterfly_encode(u):
    """蝶形编码（对应 u * F^⊗n）"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    for _ in range(int(np.log2(N))):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u * B_N * F^⊗n
    """
    x = _butterfly_encode(u)
    return x[_bit_rev_indices(len(x))]


def polar_encode_natural(u):
    """自然序编码 x = u * F^⊗n（译码器内部使用）"""
    return _butterfly_encode(u)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    xn = polar_encode_natural(u)
    print("natural:", xn)
