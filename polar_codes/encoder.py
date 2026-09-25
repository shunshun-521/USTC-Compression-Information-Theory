"""
极化码编码器
编码：x = u * F^{\\otimes n}（Arikan 标准蝶形）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """极化码编码，O(N log N) 蝶形 XOR"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    x = u.copy()
    stage = N
    while stage > 1:
        half = stage // 2
        for base in range(0, N, stage):
            for k in range(half):
                idx = base + k
                x[idx] ^= x[idx + half]
        stage = half
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u =", u, "-> x =", polar_encode(u))
