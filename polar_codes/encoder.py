"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（蝶形结构 O(N log N)）。

    默认输出与置换 SC 译码器配套（不做比特倒序）。
    若 apply_bit_reversal=True，则附加 B_N 置换（x = u * B_N * F^{⊗n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否对比特倒序置换

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    if apply_bit_reversal:
        rev = bit_reversal_permutation(N)
        return u[rev]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode (default):", polar_encode(u))
    print("encode (+bitrev):", polar_encode(u, apply_bit_reversal=True))
