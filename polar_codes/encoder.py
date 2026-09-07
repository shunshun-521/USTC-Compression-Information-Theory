"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，与译码器配套）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = N
    for _ in range(int(np.log2(N))):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def polar_generator_matrix(N):
    """生成 G_N（用于验证）"""
    u = np.eye(N, dtype=np.int8)
    return np.array([polar_encode(row) for row in u], dtype=np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
