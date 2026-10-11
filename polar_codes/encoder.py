"""
极化码编码器
编码：蝶形 polar transform（与 Vangala SCD 配套，不在末尾做比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：O(N log N) 蝶形 XOR（mcba1n / 5G 常用实现）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("length must be power of 2")
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        block = half
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
