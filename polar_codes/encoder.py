"""
极化码编码器
编码：x = u @ F^⊗n（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if n == 0:
        return np.array([0], dtype=np.int64)
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(bit_reversal_permutation(1 << n)[i])


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 u @ F^{\otimes n} 等价）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
