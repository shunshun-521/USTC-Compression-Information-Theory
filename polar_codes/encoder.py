"""
极化码编码器
编码：x = u * F^⊗n（蝶形结构，O(N log N)）
注：与常见 SC 译码器配套，不在此处施加比特倒序（B_N）；等效于 G_N = B_N F^{\otimes n} 时由译码端处理。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    perm = np.arange(N, dtype=int)
    for i in range(N):
        rev = 0
        x = i
        for _ in range(n):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        perm[i] = rev
    return perm


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 F^⊗n 对应）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode", u, "->", polar_encode(u))
