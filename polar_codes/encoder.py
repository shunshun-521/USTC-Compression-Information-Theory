"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(bin(i)[2:].zfill(n)[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 Vangala SCD 译码器配套）。

    采用分块蝶形：从块长 N 开始，每轮将左半与右半做 XOR。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
