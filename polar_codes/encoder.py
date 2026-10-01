"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(int)
    rev = rev[:, ::-1]
    powers = 2 ** np.arange(n)
    return (rev * powers).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 2 ** (stage + 1)
        half = step // 2
        for i in range(0, N, step):
            u[i:i + half] ^= u[i + half:i + step]
    br = bit_reversal_permutation(N)
    return u[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
