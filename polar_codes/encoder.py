"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗n}，O(N log N) 蝶形实现
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换 B_N）。

    蝶形：u[j] ^= u[j + step]（左支 XOR），最后 x = u[B_N]。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for left in range(0, N, 2 * step):
            right = left + step
            u[left:right] ^= u[right : right + step]
        step <<= 1

    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_decode_channel_order(llr_ch):
    """将信道序 LLR 转为 SC 因子图自然序（比特倒序）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u =", u, "-> x =", polar_encode(u))
