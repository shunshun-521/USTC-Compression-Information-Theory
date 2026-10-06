"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i: i + step] ^= u[i + step: i + 2 * step]
        step <<= 1

    br = bit_reversal_permutation(N)
    x = u[br]
    return x.astype(int)


def build_generator_matrix(N):
    """用于验证的显式生成矩阵 G = B_N F^{otimes n} mod 2"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=np.int8)[br]
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = build_generator_matrix(4)
    x2 = (u @ G) % 2
    print("matrix:", x2)
