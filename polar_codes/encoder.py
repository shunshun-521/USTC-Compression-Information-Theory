"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：x[i] = u[perm[i]]"""
    n = int(np.log2(N))
    perm = np.arange(N, dtype=int)
    for i in range(N):
        rev = 0
        for b in range(n):
            if (i >> b) & 1:
                rev |= 1 << (n - 1 - b)
        perm[i] = rev
    return perm


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half
    perm = bit_reversal_permutation(N)
    return u[perm].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
