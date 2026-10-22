"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.empty(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            r |= ((i >> b) & 1) << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：x = u @ G_N (GF(2))，G_N = F^{⊗ n}。
    与标准 SC/SCL 译码器配套（不在此处对比特倒序）。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")

    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step //= 2
    return u


def polar_encode_matrix(N):
    """生成 N×N 极化码生成矩阵 G_N = F^{⊗ n}。"""
    G = np.array([[1, 0], [1, 1]], dtype=np.int8)
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, np.array([[1, 0], [1, 1]], dtype=np.int8))
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode:", polar_encode(u))
