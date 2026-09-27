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
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br]


def build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n} for verification."""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = build_generator_matrix(4)
    print("u @ G % 2:", (u @ G) % 2)
