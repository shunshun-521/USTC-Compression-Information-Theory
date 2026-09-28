"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}，O(N log N) 蝶形结构
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    u[j] 为自然序源比特，输出 x 满足 x = u * B_N * F^{\otimes n}（GF(2)）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")

    for s in range(n):
        step = 1 << s
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    br = bit_reversal_permutation(N)
    return u[br].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
