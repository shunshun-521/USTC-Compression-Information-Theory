"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    step = 1
    while step < len(u):
        for i in range(0, len(u), 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    rev = bit_reversal_permutation(len(u))
    return u[rev]


def reorder_llr_for_decode(llr):
    """将信道 LLR 重排为译码器内部顺序（比特倒序）。译码器内部会自动调用。"""
    N = len(llr)
    return llr[bit_reversal_permutation(N)]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
